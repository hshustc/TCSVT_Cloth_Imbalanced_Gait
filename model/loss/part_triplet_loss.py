import torch
import torch.nn as nn
import torch.nn.functional as F

class PartTripletLoss(nn.Module):
    def __init__(self, margin, hard_mining=False, nonzero=True):
        super(PartTripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.nonzero = nonzero

    def forward(self, feature, label, label_type=None, label_mark=None):
        # feature: [n, m, d], label: [n, m]
        # label>=0, label_type>=0 : real
        # label>=0, label_type<0  : fake
        # label<0,  label_type<0  : pad
        # if label_type  is not None:
        #     print('feature size={}, label size={}, label_type size={}'.format(feature.size(), label.size(), label_type.size()))
        # else:
        #     print('feature size={}, label size={}'.format(feature.size(), label.size()))
        
        '''
        # remove those with label < 0 (pad for DDP)
        n, _, d = feature.size()
        mask = (label >= 0).view_as(label)
        label = label[mask].view(n, -1)
        if label_type is not None:
            label_type = label_type[mask].view(n, -1)
        if label_mark is not None:
            label_mark = label_mark[mask].view(n, -1)
        feature = feature[mask.unsqueeze(-1).repeat(1, 1, d)].view(n, -1, d)
        assert(feature.size(1) == label.size(1))
        '''
        
        # mask
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(2) == label.unsqueeze(1)).bool() # n x m x m, anchor/positive
        hn_mask = (label.unsqueeze(2) != label.unsqueeze(1)).bool() # n x m x m, anchor/negative
        # if label_type is not None and label_mark is not None:
        #     # real mask
        #     real_mask1 = (label_type.unsqueeze(2).repeat(1, 1, m) >= 0).bool() # n x m x m, real seq as anchor
        #     real_mask2 = (label_type.unsqueeze(1).repeat(1, m, 1) >= 0).bool() # n x m x m, real seq as positve/negative
        #     # anchor-positive
        #     mark_mask = (label_mark.unsqueeze(2) == label_mark.unsqueeze(1)).bool() # n x m x m, with identical label mark
        #     hp_mask = hp_mask & real_mask1 & (real_mask2 | mark_mask) # n x m x m, real seq as anchor/positive or with identical label mark
        #     # anchor-negative
        #     hn_mask = hn_mask & real_mask1 & real_mask2 # n x m x m, real seq as anchor/negative
        
        dist = self.batch_dist(feature) # n x m x m
        full_loss_metric_list = []
        nonzero_num_list = []
        for i in range(m):
            # if label_type is not None:
            #     if label_type[:, i].sum() < 0:
            #         continue
            #############################################################
            # pos_cnt = (hp_mask[:, i, :] != 0).sum(-1)
            # neg_cnt = (hn_mask[:, i, :] != 0).sum(-1)
            # print("total={}, index={}, pos_cnt={}, neg_cnt={}".format(m, i, pos_cnt, neg_cnt))
            #############################################################
            full_hp_dist = torch.masked_select(dist[:, i, :], hp_mask[:, i, :]).view(n, -1).unsqueeze(-1)
            full_hn_dist = torch.masked_select(dist[:, i, :], hn_mask[:, i, :]).view(n, -1).unsqueeze(-2)
            full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
            full_loss_metric_list.append(full_loss_metric.sum(1, keepdim=True))
            nonzero_num_list.append((full_loss_metric != 0).sum(1, keepdim=True).float())
        full_loss_metric_list = torch.cat(full_loss_metric_list, dim=1)
        nonzero_num_list = torch.cat(nonzero_num_list, dim=1)
        full_loss_metric = full_loss_metric_list.sum(1)
        nonzero_num = nonzero_num_list.sum(1)
        full_loss_metric_mean = full_loss_metric / nonzero_num
        full_loss_metric_mean[nonzero_num == 0] = 0

        # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
        return full_loss_metric_mean.mean(), nonzero_num.mean()
        '''
        dist = self.batch_dist(feature)
        dist = dist.view(-1)
        if self.hard_mining:
            # hard
            hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
            hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
            if self.margin > 0:
                hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)
            else:
                hard_loss_metric = F.softplus(hard_hp_dist - hard_hn_dist).view(n, -1)
                
            nonzero_num = (hard_loss_metric != 0).sum(1).float()

            if self.nonzero:
                hard_loss_metric_mean = hard_loss_metric.sum(1) / nonzero_num
                hard_loss_metric_mean[nonzero_num == 0] = 0
            else:
                hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)

            return hard_loss_metric_mean.mean(), nonzero_num.mean()
        else:
            # full
            full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
            full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
            if self.margin > 0:
                full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)
            else:
                full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(n, -1)  

            nonzero_num = (full_loss_metric != 0).sum(1).float()

            if self.nonzero:
                full_loss_metric_mean = full_loss_metric.sum(1) / nonzero_num
                full_loss_metric_mean[nonzero_num == 0] = 0
            else:
                full_loss_metric_mean = full_loss_metric.mean(1)
            
            # print("full_loss_metric={}, nonzero_num={}".format(full_loss_metric_mean, nonzero_num))
            return full_loss_metric_mean.mean(), nonzero_num.mean()
        '''

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist