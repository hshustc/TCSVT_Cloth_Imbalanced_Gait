import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2

class DataSet(tordata.Dataset):
    def __init__(self, seq_dir_list, seq_label_list, index_dict, resolution, cut_padding, head_label_set=[], tail_label_set=[]):
        self.seq_dir_list = seq_dir_list
        self.seq_label_list = seq_label_list
        #############################################################
        self.seq_type_list = [tmp.split('/')[-2] for tmp in self.seq_dir_list]
        self.seq_view_list = [tmp.split('/')[-1] for tmp in self.seq_dir_list]
        #############################################################
        self.index_dict = index_dict
        self.resolution = int(resolution)
        self.cut_padding = int(cut_padding)
        self.data_size = len(self.seq_label_list)
        #############################################################
        self.type_set = sorted(list(set(self.seq_type_list)))
        self.view_set = sorted(list(set(self.seq_view_list)))
        #############################################################
        self.label_set = sorted(list(set(self.seq_label_list)))
        self.head_label_set = sorted(head_label_set)
        self.tail_label_set = sorted(tail_label_set)
        assert(len( set(self.head_label_set).intersection(set(self.tail_label_set)) )==0)
        if len(self.head_label_set) > 0 or len(self.tail_label_set) > 0:
            assert(len(self.head_label_set) + len(self.tail_label_set) == len(self.label_set))
        else:
            self.head_label_set = self.label_set.copy()
        #############################################################
        self.cc_label_set = None
        self.cc_label_cnt = None
        self.cc_label_center = None
        #############################################################
        print("####################################################")
        print('DataSet Initialization')
        print('type_set={}, num={}'.format(self.type_set, len(self.type_set)))
        print('view_set={}, num={}'.format(self.view_set, len(self.view_set)))
        print('label_set={}, num={}'.format(self.label_set, len(self.label_set)))
        print('head_label_set={}, num={}'.format(self.head_label_set, len(self.head_label_set)))
        print('tail_label_set={}, num={}'.format(self.tail_label_set, len(self.tail_label_set)))
        print("####################################################")

    def __loader__(self, path):
        if self.cut_padding > 0:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0
        else: 
            return self.img2xarray(
                path).astype(
                'float32') / 255.0

    def __getitem__(self, index):
        seq_path = self.seq_dir_list[index]
        seq_imgs = self.__loader__(seq_path)
        seq_label = self.seq_label_list[index]
        seq_type = self.seq_type_list[index]
        seq_view = self.seq_view_list[index]
        return seq_imgs, '{}/{}/{}'.format(seq_label, seq_type, seq_view)

    def img2xarray(self, file_path):
        pkl_name = '{}.pkl'.format(os.path.basename(file_path))
        all_imgs = pickle.load(open(osp.join(file_path, pkl_name), 'rb'))
        return all_imgs

    def __len__(self):
        return self.data_size
