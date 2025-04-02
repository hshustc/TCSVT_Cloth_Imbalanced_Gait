import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return tensors_gather

class ReSyncBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True, DDP=True):
        super(ReSyncBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine =  affine
        self.DDP = DDP
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def extra_repr(self):
        return 'num_features={}, eps={}, momentum={}, affine={}, DDP={}'.format( \
                    self.num_features, self.eps, self.momentum, self.affine, self.DDP)

    def forward(self, x, label=None, label_type=None):
        # input: N x C, label: N, label_type: N
        # output: N x C
        # label>=0, label_type>=0 : real
        # label>=0, label_type<0  : fake
        # label<0,  label_type<0  : pad
        if self.training:
            if self.DDP:
                all_feature = torch.cat(GatherLayer.apply(x.float()), dim=0)
                if label is not None and label_type is not None: 
                    all_label = torch.cat(all_gather(label.float()), dim=0)
                    all_label_type = torch.cat(all_gather(label_type.float()), dim=0)
            else:
                all_feature = x
                all_label = label
                all_label_type = label_type

            bs, d = all_feature.size()
            if label is not None and label_type is not None: 
                real_fake_mask = (all_label.detach()>=0).view(-1, 1).repeat(1, d)
                real_mask = (all_label_type.detach()>=0).view(-1, 1).repeat(1, d)
                mask_feature = all_feature[real_fake_mask].view(-1, d)
            else:
                mask_feature = all_feature
            mask_bs = mask_feature.size(0)
            mean = torch.mean(mask_feature, dim=0)
            var = torch.var(mask_feature, dim=0, unbiased=False)
            # print('bs={}, mask_bs={}'.format(bs, mask_bs))

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.detach()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.detach() * (mask_bs*1.0/(mask_bs-1))
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x = x * self.weight.view(1, self.num_features) + self.bias.view(1, self.num_features)
        
        return x





    