import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import numpy as np
from copy import deepcopy
from .basic_blocks import *
from .sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d

class GaitSet(nn.Module):
    def __init__(self, config):
        super(GaitSet, self).__init__()
        self.config = deepcopy(config)
        assert(self.config['less_channels'] is False or self.config['more_channels'] is False)
        if self.config['more_channels']:
            self.config.update({'channels':[64, 128, 256, 512]})
        elif self.config['less_channels']:
            self.config.update({'channels':[16, 32, 64]})
        else:
            self.config.update({'channels':[32, 64, 128]})
        print("############################")
        print("GaitSet: channels={}, bin_num={}, hidden_dim={}".format(\
                self.config['channels'], self.config['bin_num'], self.config['hidden_dim']))       
        print("############################")

        self.bin_num = list(self.config['bin_num'])
        self.hidden_dim = self.config['hidden_dim']

        self.config.update({'in_channels':1})
        self.layer1 = BasicConv2d(self.config['in_channels'], self.config['channels'][0], kernel_size=5, stride=1, padding=2)
        self.layer2 = BasicConv2d(self.config['channels'][0], self.config['channels'][0], kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.layer3 = BasicConv2d(self.config['channels'][0], self.config['channels'][1], kernel_size=3, stride=1, padding=1)
        self.layer4 = BasicConv2d(self.config['channels'][1], self.config['channels'][1], kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.layer5 = BasicConv2d(self.config['channels'][1], self.config['channels'][2], kernel_size=3, stride=1, padding=1)
        self.layer6 = BasicConv2d(self.config['channels'][2], self.config['channels'][2], kernel_size=3, stride=1, padding=1)
        if len(self.config['channels']) > 3:
            self.layer7 = BasicConv2d(self.config['channels'][2], self.config['channels'][3], kernel_size=3, stride=1, padding=1)
            self.layer8 = BasicConv2d(self.config['channels'][3], self.config['channels'][3], kernel_size=3, stride=1, padding=1)            

        self.fc_bin = Part_FC(sum(self.bin_num), self.config['channels'][-1], self.hidden_dim, init='xavier')

        if self.config['cc_aug']:
            self.cc_norm = Part_Norm(sum(self.bin_num), self.hidden_dim, norm_type='BatchNorm', DDP=self.config['DDP'])
            self.cc_bin = Part_FC(sum(self.bin_num), self.hidden_dim, self.hidden_dim, init='xavier')
                
        #initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm1d, \
                        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.LayerNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_pool(self, x, n, s, batch_frames=None):
        if batch_frames is None:
            _, c, h, w = x.size()
            return torch.max(x.view(n, s, c, h, w), 1)[0]
        else:
            tmp = []
            for i in range(len(batch_frames) - 1):
                tmp.append(torch.max(x[batch_frames[i]:batch_frames[i+1], :, :, :], 0, keepdim=True)[0])
            return torch.cat(tmp, 0)

    def forward(self, silho, batch_frames=None, label=None, label_type=None, label_mark=None, cc_label_cnt=None, cc_label_center=None):
        with autocast(enabled=self.config['AMP']):
            # n: batch_size, s: frame_num, k: keypoints_num, c: channel
            if batch_frames is not None:
                batch_frames = batch_frames[0].data.cpu().numpy().tolist()
                num_seqs = len(batch_frames)
                for i in range(len(batch_frames)):
                    if batch_frames[-(i + 1)] > 0:
                        break
                    else:
                        num_seqs -= 1
                batch_frames = batch_frames[:num_seqs]
                frame_sum = np.sum(batch_frames)
                if frame_sum < silho.size(1):
                    silho = silho[:, :frame_sum, :, :]
                batch_frames = [0] + np.cumsum(batch_frames).tolist()
            x = silho.unsqueeze(2)
            del silho

            n1, s, c, h, w = x.size()
            x = self.layer1(x.view(-1, c, h, w))
            x = self.layer2(x)
            x = self.max_pool1(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.max_pool2(x)
            x = self.layer5(x)
            x = self.layer6(x)
            if len(self.config['channels']) > 3:
                x = self.layer7(x)
                x = self.layer8(x)
            
            x = self.set_pool(x, n1, s, batch_frames)

            feature = list()
            offset = 0
            for num_bin in self.bin_num:
                n2, c, h, w = x.size()
                z = x.view(n2, c, num_bin, -1).max(-1)[0] + x.view(n2, c, num_bin, -1).mean(-1)
                feature.append(z)

            feature = torch.cat(feature, dim=-1)                        # n x c x num_parts
            feature = feature.permute(2, 0, 1).contiguous()             # num_parts x n x c 
            feature = self.fc_bin(feature)                              # num_parts x n x hidden_dim
            encoder_feature = feature.permute(1, 0, 2).contiguous()     # n x num_parts x hidden_dim

            #############################################################
            cc_label = torch.empty(1).cuda() if label is None else torch.clone(label)
            cc_label_type = torch.empty(1).cuda() if label_type is None else torch.clone(label_type)
            cc_label_mark = torch.empty(1).cuda() if label_mark is None else torch.clone(label_mark)
            if self.training and self.config['cc_aug']:
                assert(cc_label_cnt.size(0) == 1 and cc_label_center.size(0) == 1)
                cc_label_cnt = cc_label_cnt.squeeze(0).detach() # n_cc 
                cc_label_center = cc_label_center.squeeze(0).detach() # n_cc x max_center x num_parts x hidden_dim
                assert(cc_label_cnt.size(0) == cc_label_center.size(0))
                num_seq = encoder_feature.size(0)
                num_cc = cc_label_cnt.size(0) 
                cc_index_per_seq = torch.randint(0, num_cc, (num_seq, self.config['cc_k'])) # n x cc_k

                fake_feature = list()
                fake_label = list()
                fake_label_type = list()
                fake_label_mark = list()
                for i in range(num_seq):
                    if label_type[i] == 0:
                        seq_feature_i = encoder_feature[i].unsqueeze(0) # 1 x num_parts x hidden_dim
                        seq_diff_i = torch.zeros_like(seq_feature_i) # 1 x num_parts x hidden_dim
                        seq_norm_i = 0.0
                        for j in range(self.config['cc_k']):
                            num_center_ij = cc_label_cnt[cc_index_per_seq[i][j]]
                            cc_center_ij = cc_label_center[cc_index_per_seq[i][j]][0:num_center_ij] # num_center x num_parts x hidden_dim
                            rand_index = torch.randperm(num_center_ij).cuda()
                            cc_diff_ij = cc_center_ij[rand_index[0]] - cc_center_ij[rand_index[1]] # num_parts x hidden_dim
                            cc_coeff_ij = torch.rand(1).cuda() * self.config['cc_s'] # 1
                            seq_norm_i = seq_norm_i + cc_coeff_ij
                            seq_diff_i = seq_diff_i + cc_coeff_ij * cc_diff_ij.unsqueeze(0) # 1 x num_parts x hidden_dim
                        if self.config['cc_normalize']:
                            seq_diff_i = (seq_diff_i/seq_norm_i).detach()
                        else:
                            seq_diff_i = seq_diff_i.detach()
                        fake_feature.append(seq_feature_i + seq_diff_i)
                        fake_label.append(label[i])
                        fake_label_type.append(-1*torch.ones_like(label_type[i]))
                        fake_label_mark.append(label_mark[i])
                fake_feature = torch.cat(fake_feature, dim=0) # n_syn x num_parts x hidden_dim
                fake_label = torch.stack(fake_label, dim=0) # n_syn
                fake_label_type = torch.stack(fake_label_type, dim=0) # n_syn
                fake_label_mark = torch.stack(fake_label_mark, dim=0) # n_syn
                feature = torch.cat((feature, fake_feature.permute(1, 0, 2).contiguous()), dim=1) # num_parts x (n + n_syn) x hidden_dim
                cc_label = torch.cat((label, fake_label), dim=0) # (n + n_syn)
                cc_label_type = torch.cat((label_type, fake_label_type), dim=0) # (n + n_syn)
                cc_label_mark = torch.cat((label_mark, fake_label_mark), dim=0) # (n + n_syn)
                # print('fake_feature size={}, fake_label size={}, fake_label_type size={}, fake_label_mark size={}, fake_label={}, fake_label_type={}, fake_label_mark={}'.format(\
                #             fake_feature.size(), fake_label.size(), fake_label_type.size(), fake_label_mark.size(), fake_label, fake_label_type, fake_label_mark))
                # print('cc_feature size={}, cc_label size={}, cc_label_type size={}, cc_label_mark size={}, cc_label={}, cc_label_type={}, cc_label_mark={}'.format(\
                #             feature.size(), cc_label.size(), cc_label_type.size(), cc_label_mark.size(), cc_label, cc_label_type, cc_label_mark))
            #############################################################            

            if label is not None:
                assert(feature.size(1) == cc_label.size(0))
            if label_type is not None:
                assert(feature.size(1) == cc_label_type.size(0))
            if label_mark is not None:
                assert(feature.size(1) == cc_label_mark.size(0))
            if self.config['cc_aug']:
                if label is not None and label_type is not None:
                    feature = self.cc_bin(F.relu(self.cc_norm(feature, cc_label, cc_label_type))) # num_parts x n x hidden_dim
                else:
                    feature = self.cc_bin(F.relu(self.cc_norm(feature, None, None))) # num_parts x n x hidden_dim
                cc_feature = feature.permute(1, 0, 2).contiguous() # n x num_parts x hidden_dim
            else:
                cc_feature = torch.empty(1).cuda()
            
            if self.config['encoder_triplet_weight'] <= 0:
                feature = feature.detach()
            if self.config['cc_triplet_weight'] <= 0:
                cc_feature = cc_feature.detach()

            return encoder_feature, cc_feature, cc_label.detach(), cc_label_type.detach(), cc_label_mark.detach()
