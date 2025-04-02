import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import copy
from .sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d
from .norm_blocks import ReSyncBatchNorm1d

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Part_Norm(nn.Module):
    def __init__(self, num_parts, channels, norm_type='BatchNorm', DDP=True):
        super(Part_Norm, self).__init__()
        assert(norm_type == 'ReBatchNorm' or norm_type == 'BatchNorm' or norm_type == 'LayerNorm')
        self.num_parts = num_parts
        self.channels = channels
        self.norm_type = norm_type
        self.DDP = DDP
        #############################################################
        if self.norm_type == 'BatchNorm':
            if self.DDP:
                part_bn = nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm1d(self.channels))
            else:
                part_bn = SynchronizedBatchNorm1d(self.channels)
        elif self.norm_type == 'ReBatchNorm':
            part_bn = ReSyncBatchNorm1d(self.channels, DDP=self.DDP)
        elif self.norm_type == 'LayerNorm':
            part_bn = nn.LayerNorm(self.channels)
        self.part_bn = clones(part_bn, self.num_parts)
        #############################################################
        self.init_weight()
    
    def init_weight(self):
        for part_bn in self.part_bn:
            if part_bn.weight is not None:
                nn.init.normal_(part_bn.weight, 1.0, 0.02)
            if part_bn.bias is not None:
                nn.init.constant_(part_bn.bias, 0.0)

    def extra_repr(self):
        return 'num_parts={}, channels={}, norm_type={}, DDP={}'.format(self.num_parts, self.channels, self.norm_type, self.DDP) 
        
    def forward(self, x, label=None, label_type=None):
        # input: num_parts x batch_size x channels, label/label_type: batch_size
        # output: num_parts x batch_size x channels
        out = list()
        for part_x, part_bn in zip(x.split(1, 0), self.part_bn):
            if self.norm_type == 'ReBatchNorm':
                out.append(part_bn(part_x.squeeze(0), label, label_type).unsqueeze(0))
            else:
                out.append(part_bn(part_x.squeeze(0)).unsqueeze(0))
        out = torch.cat(out, 0)    
        return out

class Part_FC(nn.Module):
    def __init__(self, num_parts, in_channels, out_channels, init='xavier'):
        super(Part_FC, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init = init
        assert(self.init == 'xavier' or self.init == 'identity')
        if self.init == 'xavier':
            self.part_fc = nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.zeros(num_parts, in_channels, out_channels)))
        elif self.init == 'identity':
            self.part_fc = nn.Parameter(
                        torch.zeros(num_parts, in_channels, out_channels))
            for i in range(num_parts):
                nn.init.eye_(self.part_fc[i])

    def extra_repr(self):
        return 'num_parts={}, in_channels={}, out_channels={}, init={}'.format(self.num_parts, self.in_channels, self.out_channels, self.init) 
        
    def forward(self, x):
        # input: num_parts x batch_size x in_channels, output: num_parts x batch_size x out_channels
        out = x.matmul(self.part_fc)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out