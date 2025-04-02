import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
from copy import deepcopy
import pickle

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import TripletSampler, DistributedTripletSampler, build_data_transforms
from .loss import DistributedLossWrapper, DistributedLossWrapperWithLabelType, DistributedLossWrapperWithTypeMark, all_gather
from .loss import PartTripletLoss, CenterLoss, CrossEntropyLabelSmooth
from .solver import WarmupMultiStepLR
from .network import GaitSet
from .network.sync_batchnorm import DataParallelWithCallback

class Model:
    def __init__(self, config):
        self.config = deepcopy(config)
        if self.config['DDP']:
            torch.cuda.set_device(self.config['local_rank'])
            dist.init_process_group(backend='nccl')
            self.config['cc_triplet_weight'] *= dist.get_world_size()
            self.config['encoder_triplet_weight'] *= dist.get_world_size()
            self.random_seed = self.config['random_seed'] + dist.get_rank()
        else:
            self.random_seed = self.config['random_seed']
        
        self.config.update({'num_id': len(self.config['train_source'].label_set)})
        self.encoder = GaitSet(self.config).float().cuda()
        if self.config['DDP']:
            self.encoder = DDP(self.encoder, device_ids=[self.config['local_rank']], output_device=self.config['local_rank'], find_unused_parameters=True)
        else:
            self.encoder = DataParallelWithCallback(self.encoder)
        self.build_data()
        self.build_loss()
        self.build_loss_metric()
        self.build_optimizer()

        if self.config['DDP']:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def build_data(self):
        # data augment
        if self.config['dataset_augment']:
            self.data_transforms = build_data_transforms(random_erasing=True, random_rotate=False, \
                                        random_horizontal_flip=False, random_pad_crop=False, \
                                        resolution=self.config['resolution'], random_seed=self.random_seed) 
        
        #triplet sampler
        if self.config['DDP']:
            self.triplet_sampler = DistributedTripletSampler(self.config['train_source'], \
                                        self.config['head_batch_size'], self.config['tail_batch_size'], random_seed=self.random_seed)
        else:
            self.triplet_sampler = TripletSampler(self.config['train_source'], self.config['head_batch_size'], self.config['tail_batch_size'])

    def offupdate_cluster_labels(self, type_list):
        dataset = self.config['dataset'][0].replace('-', '_')
        assert(dataset == 'CASIA_B' or dataset == 'Outdoor_Gait')
        cluster_labels = list()
        if dataset == 'CASIA_B':
            for _type in type_list:
                assert('nm' in _type or 'bg' in _type or 'cl' in _type)
                if _type in ['cl-01', 'cl-02']:
                    cluster_labels.append(1)
                else:
                    cluster_labels.append(0)
            num_clothes = len(np.unique(cluster_labels))
            max_clothes = 2
        elif dataset == 'Outdoor_Gait':
            for _type in type_list:
                assert('_nm_' in _type or '_bg_' in _type or '_cl_' in _type)
                if '_cl_' in _type:
                    cluster_labels.append(1)
                else:
                    cluster_labels.append(0)
            num_clothes = len(np.unique(cluster_labels))
            max_clothes = 2            
        return np.asarray(cluster_labels), num_clothes, max_clothes

    def onupdate_cluster_labels(self, type_list):
        dataset = self.config['dataset'][0].replace('-', '_')
        assert(dataset == 'CASIA_B' or dataset == 'Outdoor_Gait')
        cluster_labels = list()
        if dataset == 'CASIA_B':
            for _type in type_list:
                assert('nm' in _type or 'bg' in _type or 'cl' in _type)
                if _type in ['cl-01', 'cl-02']:
                    cluster_labels.append(1)
                else:
                    cluster_labels.append(0)
        elif dataset == 'Outdoor_Gait':
            for _type in type_list:
                assert('_nm_' in _type or '_bg_' in _type or '_cl_' in _type)
                if '_cl_' in _type:
                    cluster_labels.append(1)
                else:
                    cluster_labels.append(0)
        return torch.from_numpy(np.asarray(cluster_labels)).cuda()
    
    def offupdate_class_center(self, cache=False):
        #############################################################
        print('offline update class center starts')
        self.encoder.eval()
        #############################################################
        # feature_list, view_list, type_list, label_list: array
        cache_file = '{}-{:0>5}-offupdate-trainset-feature.pkl'.format(self.config['save_name'], self.config['restore_iter'])
        if cache and osp.exists(cache_file):
            print('{} EXISTS'.format(cache_file))
            feature_list, view_list, type_list, label_list = pickle.load(open(cache_file, 'rb'))
        else:
            # forward
            source = self.config['train_source']
            data_loader = tordata.DataLoader(
                dataset=source,
                batch_size=self.config['cc_offupdate_batch_size'],
                sampler=tordata.sampler.SequentialSampler(source),
                collate_fn=self.collate_fn,
                num_workers=self.config['num_workers'])
            feature_list = list()
            view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
            type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
            label_list = list()
            for i, x in enumerate(data_loader):
                seq, label_type_view, batch_frame = x
                label = [tmp.split('/')[0] for tmp in label_type_view]
                seq = self.np2var(seq).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()
                output = self.encoder(seq, batch_frame)
                feature = output[self.config['cc_offupdate_feat_idx']]
                feature_list.append(feature.data.cpu().numpy())             
                label_list += label
            feature_list = np.concatenate(feature_list, 0)
            view_list = np.asarray(view_list)
            type_list = np.asarray(type_list)
            label_list = np.asarray(label_list)
            if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                pickle.dump([feature_list, view_list, type_list, label_list], open(cache_file, 'wb'), protocol=4)
                print('{} SAVED'.format(cache_file))
        #############################################################
        # cluster
        cc_label_set = list()
        cc_label_cnt = list()
        cc_label_center = list()
        for label in sorted(self.config['train_source'].label_set):
            label_index = np.where(label_list == label)[0]
            label_view = view_list[label_index]
            label_type = type_list[label_index]
            label_feature = feature_list[label_index]
            cluster_labels, num_clothes, max_clothes = self.offupdate_cluster_labels(label_type)
            if num_clothes > 1:
                label_center = list()
                for clulabel in sorted(np.unique(cluster_labels)):
                    clulabel_index = np.where(cluster_labels == clulabel)[0]
                    clulabel_feature = label_feature[clulabel_index]
                    mean_clulabel_feature = np.mean(clulabel_feature, axis=0, keepdims=True)
                    label_center.append(mean_clulabel_feature)
                    # print('label={}, clulabel={}, clulabel_feature size={}'.format(label, clulabel, clulabel_feature.shape))
                if num_clothes < max_clothes:
                    for i in range(max_clothes - num_clothes):
                        label_center.append(np.zeros_like(mean_clulabel_feature))
                label_center = np.concatenate(label_center, axis=0)
                cc_label_set.append(label)
                cc_label_cnt.append(num_clothes)
                cc_label_center.append(np.expand_dims(label_center, axis=0))
                # print('label={}, label_center size={}'.format(label, label_center.shape))
        cc_label_cnt = np.asarray(cc_label_cnt)
        cc_label_center = np.concatenate(cc_label_center, axis=0)
        # print('cc_label_center size={}'.format(cc_label_center.shape))
        self.config['train_source'].cc_label_set = cc_label_set
        self.config['train_source'].cc_label_cnt = torch.from_numpy(cc_label_cnt).cuda()
        self.config['train_source'].cc_label_center = torch.from_numpy(cc_label_center).cuda()
        #############################################################
        # all_gather
        if self.config['DDP']:
            gather_cc_label_center = torch.stack(all_gather(self.config['train_source'].cc_label_center), dim=0)
            print('rank={}, gather_cc_label_center size={}'.format(dist.get_rank(), gather_cc_label_center.size()))
            self.config['train_source'].cc_label_center = torch.mean(gather_cc_label_center, dim=0)
        #############################################################
        self.encoder.train()
        print('offline update class center ends')
        #############################################################

    def onupdate_class_center(self, feature, label, seq_type, seq_view, train_label_set, train_type_set, train_view_set):
        #############################################################
        # print('online update class center starts')
        #############################################################
        if self.config['DDP']:
            feature = torch.cat(all_gather(feature), dim=0)
            label = torch.cat(all_gather(label), dim=0)
            seq_type = torch.cat(all_gather(seq_type), dim=0)
            seq_view = torch.cat(all_gather(seq_view), dim=0)
        feature = feature.detach()
        label = label.detach()
        seq_type = seq_type.detach()
        seq_view = seq_view.detach()
        assert(feature.size(0) == label.size(0))
        assert(feature.size(0) == seq_type.size(0))
        assert(feature.size(0) == seq_view.size(0))
        # print('feature size={}, label size={}'.format(feature.size(), label.size()))
        # print('label set={}, label set size={}'.format(torch.unique(label), torch.unique(label).size()))

        for l in sorted(torch.unique(label)):
            l_name = train_label_set[l]
            if l_name in self.config['train_source'].cc_label_set:
                l_mask = (label == l).bool()
                l_feature = feature[l_mask]
                l_type = seq_type[l_mask]
                l_type_name = [train_type_set[tmp] for tmp in l_type]
                cluster_labels = self.onupdate_cluster_labels(l_type_name)
                l_cc_index = self.config['train_source'].cc_label_set.index(l_name)
                l_cc_center = self.config['train_source'].cc_label_center[l_cc_index]
                for clulabel in sorted(torch.unique(cluster_labels)):
                    clulabel_index = (cluster_labels == clulabel).bool()
                    clulabel_feature = l_feature[clulabel_index]
                    assert(l_cc_center.ndim == clulabel_feature.ndim)
                    if clulabel_feature.size(0) >= 2:
                        mean_clulabel_feature = torch.mean(clulabel_feature, dim=0, keepdim=True)
                        l_cc_center[clulabel] = self.config['cc_onupdate_momentum'] * l_cc_center[clulabel] + \
                                                    (1-self.config['cc_onupdate_momentum']) * mean_clulabel_feature
                    # print('l_name={}, l_feature={}, l_type_name={}, cluster_labels={}, l_cc_index={}, l_cc_center={}, clulabel_feature={}, mean_clulabel_feature=={}'.format( \
                    #     l_name, l_feature.size(), l_type_name, cluster_labels, l_cc_index, l_cc_center.size(), clulabel_feature.size(), mean_clulabel_feature.size()))
                self.config['train_source'].cc_label_center[l_cc_index] = l_cc_center
        #############################################################
        # print('online update class center ends')
        #############################################################

    def build_loss(self):
        if self.config['cc_triplet_weight'] > 0:
            self.cc_triplet_loss = PartTripletLoss(self.config['cc_triplet_margin']).float().cuda()
            if self.config['DDP']:
                self.cc_triplet_loss = DistributedLossWrapperWithTypeMark(self.cc_triplet_loss, dim=1)

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss = PartTripletLoss(self.config['encoder_triplet_margin']).float().cuda()
            if self.config['DDP']:
                self.encoder_triplet_loss = DistributedLossWrapper(self.encoder_triplet_loss, dim=1)

    def build_loss_metric(self):
        if self.config['cc_triplet_weight'] > 0:
            self.cc_triplet_loss_metric = [[], []]

        if self.config['encoder_triplet_weight'] > 0:
            self.encoder_triplet_loss_metric = [[], []]

        self.total_loss_metric = []
    
    def build_optimizer(self):
        #lr and weight_decay
        base_lr = self.config['lr']
        base_weight_decay = self.config['weight_decay'] if base_lr > 0 else 0
        cc_lr = self.config['cc_lr'] if self.config['cc_lr'] is not None else base_lr
        cc_weight_decay = self.config['weight_decay'] if cc_lr > 0 else 0

        #params
        if self.config['cc_aug']:
            cc_params_id = list()
            # cc_params_id.extend(list(map(id, self.encoder.module.cc_bin.parameters())))
            for name, module in self.encoder.module.named_children():
                if 'cc_' in name:
                    print('CC Layer: {} {}'.format(name, module))
                    cc_params_id.extend(list(map(id, module.parameters())))
                else:
                    print('Base Layer: {} {}'.format(name, module))
            base_params = filter(lambda p: id(p) not in cc_params_id, self.encoder.parameters())
            cc_params = filter(lambda p: id(p) in cc_params_id, self.encoder.parameters())
            tg_params =[{'params': base_params, 'lr': base_lr, 'weight_decay': base_weight_decay}, \
                        {'params': cc_params, 'lr': cc_lr, 'weight_decay': cc_weight_decay}]
        else:
            tg_params =[{'params': self.encoder.parameters(), 'lr': base_lr, 'weight_decay': base_weight_decay}]
 
        #optimizer
        if self.config['optimizer_type'] == 'SGD':
            self.optimizer = optim.SGD(tg_params, lr=self.config['lr'], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        elif self.config['optimizer_type'] == 'ADAM': #if ADAM set the first stepsize equal to total_iter
            self.optimizer = optim.Adam(tg_params, lr=self.config['lr'])
        if self.config['warmup']:
            self.scheduler = WarmupMultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['milestones'], gamma=self.config['gamma'])

        #AMP
        if self.config['AMP']:
            self.scaler = GradScaler()

    def compute_grad_norm(self, model):
        layer_name_list = []
        layer_grad_list = []
        for layer_name, layer_params in model.named_parameters():
            layer_grad = torch.norm(layer_params.grad.detach(), 2)
            layer_name_list.append(layer_name)
            layer_grad_list.append(layer_grad)
        layer_grad_list = torch.stack(layer_grad_list)
        sorted_grad, sorted_index = torch.sort(layer_grad_list)
        max_layer_name = layer_name_list[sorted_index[-1]]
        max_layer_grad = layer_grad_list[sorted_index[-1]]
        all_layer_grad = torch.norm(layer_grad_list, 2)
        return max_layer_name, max_layer_grad, all_layer_grad

    def fit(self):
        self.encoder.train()
        if self.config['restore_iter'] > 0:
            self.load(self.config['restore_iter'])
        else:
            self.config['restore_iter'] = 0
            if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                self.save()

        train_loader = tordata.DataLoader(
            dataset=self.config['train_source'],
            batch_sampler=self.triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        train_label_set = sorted(list(self.config['train_source'].label_set))
        head_train_label_set = sorted(list(self.config['train_source'].head_label_set))
        # train_label_set.sort()
        train_type_set = sorted(list(self.config['train_source'].type_set))
        train_view_set = sorted(list(self.config['train_source'].view_set))

        if self.config['cc_aug']:
            self.offupdate_class_center(cache=True)

        _time1 = datetime.now()
        for seq, label_type_view, batch_frame in train_loader:
            #############################################################
            if self.config['cc_aug'] and self.config['cc_offupdate_interval'] > 0 and \
                self.config['restore_iter'] % self.config['cc_offupdate_interval'] == 0:
                self.offupdate_class_center(cache=True)
            #############################################################
            self.optimizer.zero_grad()

            seq = self.np2var(seq).float()
            label = [tmp.split('/')[0] for tmp in label_type_view]
            seq_type = [tmp.split('/')[1] for tmp in label_type_view]
            seq_view = [tmp.split('/')[2] for tmp in label_type_view]
            #############################################################   
            target_seq_type = [train_type_set.index(l) for l in seq_type]
            target_seq_type = self.np2var(np.asarray(target_seq_type)).long()
            target_seq_view = [train_view_set.index(l) for l in seq_view]
            target_seq_view = self.np2var(np.asarray(target_seq_view)).long()
            #############################################################
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.asarray(target_label)).long()
            target_label_type = [1 if l in head_train_label_set else 0 for l in label]
            target_label_type = self.np2var(np.asarray(target_label_type)).long()
            if self.config['DDP']:
                batch_size_per_world = int(self.triplet_sampler.batch_size_per_world)
                target_label_mark = [dist.get_rank()*batch_size_per_world+i for i in range(len(label))]
                # print('rank={}, batch_size_per_world={}, target_label_mark={}'.format(dist.get_rank(), batch_size_per_world, target_label_mark))
            else:
                target_label_mark = [i for i in range(len(label))]
                # print('target_label_mark={}'.format(target_label_mark))
            target_label_mark = self.np2var(np.asarray(target_label_mark)).long()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            #############################################################
            if self.config['cc_aug']:
                if self.config['DDP']:
                    gpu_num = 1
                else:
                    gpu_num = min(torch.cuda.device_count(), seq.size(0))
                cc_label_cnt = self.config['train_source'].cc_label_cnt
                cc_label_center = self.config['train_source'].cc_label_center
                cc_label_cnt = torch.cat(gpu_num*[cc_label_cnt.unsqueeze(0)], dim=0)
                cc_label_center = torch.cat(gpu_num*[cc_label_center.unsqueeze(0)], dim=0)
                # print("forward cc_label_cnt size={}".format(cc_label_cnt.size()))
                # print("forward cc_label_center size={}".format(cc_label_center.size()))
                #############################################################
                with autocast(enabled=self.config['AMP']):
                    encoder_feature, cc_feature, cc_label, cc_label_type, cc_label_mark = \
                        self.encoder(seq, batch_frame, label=target_label, label_type=target_label_type, label_mark=target_label_mark, \
                                        cc_label_cnt=cc_label_cnt, cc_label_center=cc_label_center)
            else:
                with autocast(enabled=self.config['AMP']):
                    encoder_feature, cc_feature, cc_label, cc_label_type, cc_label_mark = \
                        self.encoder(seq, batch_frame, label=target_label, label_type=target_label_type, label_mark=target_label_mark)
            
            if self.config['cc_onupdate_interval'] > 0 and \
                self.config['restore_iter'] % self.config['cc_onupdate_interval'] == 0:
                self.onupdate_class_center(encoder_feature.float().detach(), \
                        target_label.detach(), target_seq_type.detach(), target_seq_view.detach(), \
                        train_label_set, train_type_set, train_view_set)

            loss = torch.zeros(1).to(encoder_feature.device)

            if self.config['cc_triplet_weight'] > 0:
                assert(cc_feature.size(0) == cc_label.size(0) and cc_feature.size(0) == cc_label_type.size(0))
                cc_triplet_feature = cc_feature.float().permute(1, 0, 2).contiguous()
                cc_triplet_label = cc_label.unsqueeze(0).repeat(cc_triplet_feature.size(0), 1)
                cc_triplet_label_type = cc_label_type.unsqueeze(0).repeat(cc_triplet_feature.size(0), 1)
                cc_triplet_label_mark = cc_label_mark.unsqueeze(0).repeat(cc_triplet_feature.size(0), 1)
                triplet_loss_metric, nonzero_num = self.cc_triplet_loss(cc_triplet_feature, cc_triplet_label, cc_triplet_label_type, cc_triplet_label_mark)
                loss += triplet_loss_metric.mean() * self.config['cc_triplet_weight']
                self.cc_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.cc_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())

            if self.config['encoder_triplet_weight'] > 0:
                encoder_triplet_feature = encoder_feature.float().permute(1, 0, 2).contiguous()
                encoder_triplet_label = target_label.unsqueeze(0).repeat(encoder_triplet_feature.size(0), 1)
                triplet_loss_metric, nonzero_num = self.encoder_triplet_loss(encoder_triplet_feature, encoder_triplet_label)
                loss += triplet_loss_metric.mean() * self.config['encoder_triplet_weight']
                self.encoder_triplet_loss_metric[0].append(triplet_loss_metric.mean().data.cpu().numpy())
                self.encoder_triplet_loss_metric[1].append(nonzero_num.mean().data.cpu().numpy())

            self.total_loss_metric.append(loss.data.cpu().numpy())

            if loss > 1e-9:
                if self.config['AMP']:
                    self.scaler.scale(loss).backward()
                    if self.config['clip_grad_norm'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        max_layer_name, max_layer_grad, all_layer_grad = self.compute_grad_norm(self.encoder)
                        if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                            if self.config['restore_iter'] % 20 == 0:
                                print('before clip: iter={:0>6d}, max_layer_name={}, max_layer_grad={:.3f}, all_layer_grad={:.3f}, max_grad_ratio={:.3f}'.format(\
                                        self.config['restore_iter'], max_layer_name, max_layer_grad, all_layer_grad, max_layer_grad*1.0/all_layer_grad))
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config['clip_grad_norm'])
                        max_layer_name2, max_layer_grad2, all_layer_grad2 = self.compute_grad_norm(self.encoder)
                        if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                            if self.config['restore_iter'] % 20 == 0:
                                print('after clip: iter={:0>6d}, max_layer_name={}, max_layer_grad={:.3f}, all_layer_grad={:.3f}, max_grad_ratio={:.3f}'.format(\
                                        self.config['restore_iter'], max_layer_name2, max_layer_grad2, all_layer_grad2, max_layer_grad2*1.0/all_layer_grad2))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:  
                    loss.backward()
                    if self.config['clip_grad_norm'] > 0:
                        max_layer_name, max_layer_grad, all_layer_grad = self.compute_grad_norm(self.encoder)
                        if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                            print('before clip: iter={:0>6d}, max_layer_name={}, max_layer_grad={:.3f}, all_layer_grad={:.3f}, max_grad_ratio={:.3f}'.format(\
                                    self.config['restore_iter'], max_layer_name, max_layer_grad, all_layer_grad, max_layer_grad*1.0/all_layer_grad))
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config['clip_grad_norm'])
                        max_layer_name2, max_layer_grad2, all_layer_grad2 = self.compute_grad_norm(self.encoder)
                        if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                            print('after clip: iter={:0>6d}, max_layer_name={}, max_layer_grad={:.3f}, all_layer_grad={:.3f}, max_grad_ratio={:.3f}'.format(\
                                    self.config['restore_iter'], max_layer_name2, max_layer_grad2, all_layer_grad2, max_layer_grad2*1.0/all_layer_grad2))
                    self.optimizer.step()
                    self.scheduler.step()
                
            if self.config['restore_iter'] % 100 == 0:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    print(datetime.now() - _time1)
                    _time1 = datetime.now()
                    self.print_info()
                self.build_loss_metric()
            self.config['restore_iter'] += 1
            if self.config['restore_iter'] % 10000 == 0 or self.config['restore_iter'] == self.config['total_iter']:
                if (not self.config['DDP']) or (self.config['DDP'] and dist.get_rank() == 0):
                    self.save()
            if self.config['restore_iter'] == self.config['total_iter']:
                break

    def print_info(self):
        print('iter {}:'.format(self.config['restore_iter']))

        def print_loss_info(loss_name, loss_metric, loss_weight, loss_info):
            print('{:#^30}: loss_metric={:.6f}, loss_weight={:.6f}, {}'.format(loss_name, np.mean(loss_metric), loss_weight, loss_info))
    
        if self.config['cc_triplet_weight'] > 0:
            loss_name = 'CC Triplet'
            loss_metric = self.cc_triplet_loss_metric[0]
            loss_weight = self.config['cc_triplet_weight']
            loss_info = 'nonzero_num={:.6f}, margin={}'.format(np.mean(self.cc_triplet_loss_metric[1]), self.config['cc_triplet_margin'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        if self.config['encoder_triplet_weight'] > 0:
            loss_name = 'Encoder Triplet'
            loss_metric = self.encoder_triplet_loss_metric[0]
            loss_weight = self.config['encoder_triplet_weight']
            loss_info = 'nonzero_num={:.6f}, margin={}'.format(np.mean(self.encoder_triplet_loss_metric[1]), self.config['encoder_triplet_margin'])
            print_loss_info(loss_name, loss_metric, loss_weight, loss_info)

        print('{:#^30}: total_loss_metric={:.6f}'.format('Total Loss', np.mean(self.total_loss_metric)))
        
        #optimizer
        if self.config['cc_triplet_weight'] > 0:
            print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}, cc_lr={:.6f}, cc_weight_decay={:.6f}'.format( \
                'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay'], \
                self.optimizer.param_groups[1]['lr'], self.optimizer.param_groups[1]['weight_decay']))
        else:
            print('{:#^30}: type={}, base_lr={:.6f}, base_weight_decay={:.6f}'.format( \
                'Optimizer', self.config['optimizer_type'], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay']))
        print('{:#^30}: num_label_set={}, num_head_label_set={}, num_tail_label_set={}'.format( \
            'TrainDataSet', len(self.config['train_source'].label_set), len(self.config['train_source'].head_label_set), len(self.config['train_source'].tail_label_set))) 
        print('{:#^30}: num_label_set={}, num_head_label_set={}, num_tail_label_set={}'.format( \
            'TestDataSet', len(self.config['test_source'].label_set), len(self.config['test_source'].head_label_set), len(self.config['test_source'].tail_label_set))) 
        print('{:#^30}: pid_fname={}, head_batch_size={}, tail_batch_size={}, tail_split={}'.format( \
            'TrainDataLoader', self.config['pid_fname'], self.config['head_batch_size'], self.config['tail_batch_size'], self.config['tail_split']))
        if self.config['cc_aug']:
            print('{:#^30}: cc_k={}, cc_s={}, cc_normalize={}'.format( \
                'CC', self.config['cc_k'], self.config['cc_s'], self.config['cc_normalize']))
            print('{:#^30}: cc_offupdate_interval={}, cc_onupdate_interval={}, cc_onupdate_momentum={}'.format( \
                'CC', self.config['cc_offupdate_interval'], self.config['cc_onupdate_interval'], self.config['cc_onupdate_momentum']))           
        sys.stdout.flush()

    def transform(self, flag, batch_size=1, feat_idx=0):
        self.encoder.eval()
        source = self.config['test_source'] if flag == 'test' else self.config['train_source']
        self.config['sample_type'] = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'])

        feature_list = list()
        view_list = [tmp.split('/')[-1] for tmp in source.seq_dir_list]
        seq_type_list = [tmp.split('/')[-2] for tmp in source.seq_dir_list]
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, label_type_view, batch_frame = x
            label = [tmp.split('/')[0] for tmp in label_type_view]
            seq = self.np2var(seq).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            output = self.encoder(seq, batch_frame)
            feature = output[feat_idx]
            feature_list.append(feature.detach())             
            label_list += label

        return torch.cat(feature_list, 0), view_list, seq_type_list, label_list

    def collate_fn(self, batch):
        batch_size = len(batch)
        seqs = [batch[i][0] for i in range(batch_size)]
        label = [batch[i][1] for i in range(batch_size)]
        batch = [seqs, label, None]
        batch_frames = []
        if self.config['DDP']:
            gpu_num = 1
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)

        # generate batch_frames for next step
        for gpu_id in range(gpu_num):
            batch_frames_sub = []
            for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                if i < batch_size:
                    if self.config['sample_type'] == 'random':
                        batch_frames_sub.append(self.config['frame_num'])
                    elif self.config['sample_type'] == 'all':
                        batch_frames_sub.append(seqs[i].shape[0])
                    elif self.config['sample_type'] == 'random_fn':
                        frame_num = np.random.randint(self.config['min_frame_num'], self.config['max_frame_num'])
                        batch_frames_sub.append(frame_num)
            batch_frames.append(batch_frames_sub)
        if len(batch_frames[-1]) != batch_per_gpu:
            for i in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)

        # select frames from each seq 
        def select_frame(index):
            sample = seqs[index]
            frame_set = np.arange(sample.shape[0])
            frame_num = batch_frames[int(index / batch_per_gpu)][int(index % batch_per_gpu)]
            if len(frame_set) >= frame_num:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=False))
            else:
                frame_id_list = sorted(np.random.choice(frame_set, frame_num, replace=True))
            return sample[frame_id_list, :, :]
        seqs = list(map(select_frame, range(len(seqs))))        

        # data augmentation
        def transform_seq(index):
            sample = seqs[index]
            return self.data_transforms(sample)
        if self.config['dataset_augment']:
            seqs = list(map(transform_seq, range(len(seqs))))  

        # concatenate seqs for each gpu if necessary
        if self.config['sample_type'] == 'random':
            seqs = np.asarray(seqs)                      
        elif self.config['sample_type'] == 'all' or self.config['sample_type'] == 'random_fn':
            max_sum_frames = np.max([np.sum(batch_frames[gpu_id]) for gpu_id in range(gpu_num)])
            new_seqs = []
            for gpu_id in range(gpu_num):
                tmp = []
                for i in range(batch_per_gpu * gpu_id, batch_per_gpu * (gpu_id + 1)):
                    if i < batch_size:
                        tmp.append(seqs[i])
                tmp = np.concatenate(tmp, 0)
                tmp = np.pad(tmp, \
                    ((0, max_sum_frames - tmp.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                new_seqs.append(np.asarray(tmp))
            seqs = np.asarray(new_seqs)  

        batch[0] = seqs
        if self.config['sample_type'] == 'all' or self.config['sample_type'] == 'random_fn':
            batch[-1] = np.asarray(batch_frames)
        
        return batch

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x)) 

    def save(self):
        os.makedirs(osp.join('checkpoint', self.config['model_name']), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], self.config['restore_iter'])))
        torch.save([self.optimizer.state_dict(), self.scheduler.state_dict()],
                   osp.join('checkpoint', self.config['model_name'],
                            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], self.config['restore_iter'])))

    def load(self, restore_iter):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-encoder.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.encoder.load_state_dict(encoder_ckp)
        optimizer_ckp = torch.load(osp.join(
            'checkpoint', self.config['model_name'],
            '{}-{:0>5}-optimizer.ptm'.format(self.config['save_name'], restore_iter)), map_location=map_location)
        self.optimizer.load_state_dict(optimizer_ckp[0])
        self.scheduler.load_state_dict(optimizer_ckp[1])  

    def init_model(self, init_model):
        if self.config['DDP']:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        else:
            map_location = None
        encoder_state_dict = self.encoder.state_dict()
        ckp_state_dict = torch.load(init_model, map_location=map_location)
        init_state_dict = {k: v for k, v in ckp_state_dict.items() if k in encoder_state_dict}
        drop_state_dict = {k: v for k, v in ckp_state_dict.items() if k not in encoder_state_dict}
        print('#######################################')
        if init_state_dict:
            print("Useful Layers in Init_model for Initializaiton:\n", init_state_dict.keys())
        else:
            print("None of Layers in Init_model is Used for Initializaiton.")
        print('#######################################')
        if drop_state_dict:
            print("Useless Layers in Init_model for Initializaiton:\n", drop_state_dict.keys())
        else:
            print("All Layers in Init_model are Used for Initialization.")
        encoder_state_dict.update(init_state_dict)
        none_init_state_dict = {k: v for k, v in encoder_state_dict.items() if k not in init_state_dict}
        print('#######################################')
        if none_init_state_dict:
            print("The Layers in Target_model that Are *Not* Initialized:\n", none_init_state_dict.keys())
        else:
            print("All Layers in Target_model are Initialized")  
        print('#######################################')      
        self.encoder.load_state_dict(encoder_state_dict)    
