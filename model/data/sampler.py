import torch
import torch.utils.data as tordata
import torch.distributed as dist
import math
import random
import numpy as np

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, head_batch_size, tail_batch_size):
        self.dataset = dataset
        self.head_batch_size = head_batch_size
        self.tail_batch_size = tail_batch_size
        self.head_label_set = self.dataset.head_label_set
        self.tail_label_set = self.dataset.tail_label_set

    def __iter__(self):
        while (True):
            sample_indices = list()
            # pid_list = np.random.choice(self.dataset.label_set, 
            #     self.batch_size[0], replace=False)
            head_pid_list = np.asarray([])
            if len(self.head_label_set) > 0 and self.head_batch_size[0] > 0:
                head_pid_list = np.random.choice(self.head_label_set, self.head_batch_size[0], replace=False)
            tail_pid_list = np.asarray([])
            if len(self.tail_label_set) > 0 and self.tail_batch_size[0] > 0:
                tail_pid_list = np.random.choice(self.tail_label_set, self.tail_batch_size[0], replace=False)
            pid_list = np.concatenate((head_pid_list, tail_pid_list))
            # print('pid_list={}'.format(pid_list))
            for pid in pid_list:
                if pid in self.head_label_set:
                    batch_seqs_per_id = self.head_batch_size[1]
                elif pid in self.tail_label_set:
                    batch_seqs_per_id = self.tail_batch_size[1]
                _index = self.dataset.index_dict[pid]
                if len(_index) >= batch_seqs_per_id:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=False).tolist()
                else:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=True).tolist()             
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

def sync_random_sample_list(obj_list, k):
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]

class DistributedTripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, head_batch_size, tail_batch_size, world_size=None, rank=None, random_seed=2019, batch_shuffle=False):
        np.random.seed(random_seed)
        random.seed(random_seed)
        print("random_seed={} for DistributedTripletSampler".format(random_seed))
        
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.head_batch_size = head_batch_size
        self.tail_batch_size = tail_batch_size
        self.head_label_set = self.dataset.head_label_set
        self.tail_label_set = self.dataset.tail_label_set
        self.world_size = world_size
        self.rank = rank
        self.random_seed = 0
        self.batch_shuffle = batch_shuffle

        self.total_size = self.head_batch_size[0] * self.head_batch_size[1] + \
                            self.tail_batch_size[0] * self.tail_batch_size[1]
        assert(self.total_size % self.world_size == 0) 
        self.batch_size_per_world = int(self.total_size / self.world_size)             

    def __iter__(self):
        while (True):
            sample_indices = list()
            head_pid_list = np.asarray([])
            if len(self.head_label_set) > 0 and self.head_batch_size[0] > 0:
                head_pid_list = sync_random_sample_list(self.head_label_set, self.head_batch_size[0])
            tail_pid_list = np.asarray([])
            if len(self.tail_label_set) > 0 and self.tail_batch_size[0] > 0:
                tail_pid_list = sync_random_sample_list(self.tail_label_set, self.tail_batch_size[0])
            pid_list = np.concatenate((head_pid_list, tail_pid_list))
            # print('rank={}, pid_list={}'.format(self.rank, pid_list))
            for pid in pid_list:
                if pid in self.head_label_set:
                    batch_seqs_per_id = self.head_batch_size[1]
                elif pid in self.tail_label_set:
                    batch_seqs_per_id = self.tail_batch_size[1]
                _index = self.dataset.index_dict[pid]
                _index = sync_random_sample_list(_index, batch_seqs_per_id)        
                sample_indices += _index

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(sample_indices, len(sample_indices))
            sample_indices = sample_indices[self.rank:self.total_size:self.world_size]
            # assert(len(sample_indices) == self.batch_size_per_world)

            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

'''
class DistributedTripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, head_batch_size, tail_batch_size, world_size=None, rank=None, random_seed=2019):
        np.random.seed(random_seed)
        random.seed(random_seed)
        print("random_seed={} for DistributedTripletSampler".format(random_seed))
        
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.head_batch_size = head_batch_size
        self.tail_batch_size = tail_batch_size
        self.head_label_set = self.dataset.head_label_set
        self.tail_label_set = self.dataset.tail_label_set
        self.world_size = world_size
        self.rank = rank
        self.random_seed = 0
        
        #############################################################
        if len(self.head_label_set) > 0 and self.head_batch_size[0] > 0:
            if len(self.tail_label_set) > 0 and self.tail_batch_size[0] > 0:
                # head
                self.head_world_size = min(int(self.world_size/2), self.head_batch_size[0])
                assert(self.head_batch_size[0] % self.head_world_size == 0)
                self.head_batch_ids_per_world = int(math.ceil(self.head_batch_size[0] * 1.0 / self.head_world_size))
                self.head_total_batch_per_world = int(math.ceil(len(self.head_label_set) * 1.0 / self.head_batch_size[0]))
                # tail
                self.tail_world_size = self.world_size - self.head_world_size
                assert(self.tail_batch_size[0] % self.tail_world_size == 0)
                self.tail_batch_ids_per_world = int(math.ceil(self.tail_batch_size[0] * 1.0 / self.tail_world_size))
                self.tail_total_batch_per_world = int(math.ceil(len(self.tail_label_set) * 1.0 / self.tail_batch_size[0]))
                # total
                self.total_batch_per_world = min(self.head_total_batch_per_world, self.tail_total_batch_per_world)
                self.max_seqs_per_world = max(self.head_batch_ids_per_world*self.head_batch_size[1], self.tail_batch_ids_per_world*self.tail_batch_size[1])
            else:
                self.head_batch_ids_per_world = int(math.ceil(self.head_batch_size[0] * 1.0 / self.world_size))
                assert(self.head_batch_size[0] % self.world_size == 0)
                self.total_batch_per_world = int(math.ceil(len(self.head_label_set) * 1.0 / self.head_batch_size[0]))
                self.max_seqs_per_world = self.head_batch_ids_per_world*self.head_batch_size[1]
        else:
            self.tail_batch_ids_per_world = int(math.ceil(self.tail_batch_size[0] * 1.0 / self.world_size))
            assert(self.tail_batch_size[0] % self.world_size == 0)
            self.total_batch_per_world = int(math.ceil(len(self.tail_label_set) * 1.0 / self.tail_batch_size[0]))
            self.max_seqs_per_world = self.tail_batch_ids_per_world*self.tail_batch_size[1]
        #############################################################
                                

    def __iter__(self):
        while (True):
            g = torch.Generator()
            g.manual_seed(self.random_seed)
            if len(self.head_label_set) > 0 and self.head_batch_size[0] > 0:
                if len(self.tail_label_set) > 0 and self.tail_batch_size[0] > 0:
                    world_size = self.head_world_size
                    if self.rank < world_size:
                        label_set = self.head_label_set
                        batch_ids_per_world = self.head_batch_ids_per_world
                    else:
                        label_set = self.tail_label_set
                        batch_ids_per_world = self.tail_batch_ids_per_world
                else:
                    world_size = self.world_size
                    label_set = self.head_label_set
                    batch_ids_per_world = self.head_batch_ids_per_world
            else:
                world_size = self.world_size
                label_set = self.tail_label_set
                batch_ids_per_world = self.tail_batch_ids_per_world                    
            pid_index_all_world = torch.randperm(len(label_set), generator=g).tolist()
            pid_index_cur_world = pid_index_all_world[self.rank:len(label_set):world_size]
            # if self.rank == 0:
            #     print("random_seed={}".format(self.random_seed))
            #     print("pid_index_all_world={}, pid_index_cur_world={}".format(pid_index_all_world, pid_index_cur_world))
            #     print("batch_ids_per_world={}, total_batch_per_world={}".format(self.batch_ids_per_world, self.total_batch_per_world))
            
            sample_indices = list()
            # pid_index_cur_batch = random.sample(pid_index_cur_world, self.batch_ids_per_world)
            pid_index_cur_batch = np.random.choice(pid_index_cur_world, batch_ids_per_world, replace=False)
            # print('rank={}, pid_list={}'.format(self.rank, [label_set[tmp] for tmp in pid_index_cur_batch]))
            for pid_index in pid_index_cur_batch:
                pid_name = label_set[pid_index]
                if pid_name in self.head_label_set:
                    batch_seqs_per_id = self.head_batch_size[1]
                elif pid_name in self.tail_label_set:
                    batch_seqs_per_id = self.tail_batch_size[1]
                _index = self.dataset.index_dict[pid_name]
                # _index = random.choices(_index, k=self.batch_size[1])
                if len(_index) >= batch_seqs_per_id:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=False).tolist()
                else:
                    _index = np.random.choice(_index, batch_seqs_per_id, replace=True).tolist() 
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

    def set_random_seed(self, seed):
        self.random_seed = seed
'''