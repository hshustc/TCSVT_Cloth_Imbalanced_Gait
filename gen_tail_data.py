#!/usr/bin/env python
# coding=utf-8
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import argparse

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CASIA_B', choices=['CASIA_B', 'OUMVLP_CL', 'Outdoor_Gait'], type=str, help='name of dataset')
parser.add_argument('--dataset_path', default='/mnt/Dataset/casia_b/', type=str, help='path to dataset')
parser.add_argument('--resolution', default=128, type=int, help='resolution')
parser.add_argument('--tail_num', default=0.5, type=int, help='split train for tail')
parser.add_argument('--shuffle', default=True, type=boolean_string, help='shuffle dataset or not')
parser.add_argument('--seed', default=2022, type=int, help='random seed')

args = parser.parse_args()
if args.dataset_path[-1] == '/':
    args.dataset_path = args.dataset_path[:-1]
print("Args:", args)

if args.shuffle:
    suffix = 'tn{:0>2d}_seed{}'.format(args.tail_num, args.seed)
else:
    suffix = 'tn{:0>2d}_noshuffle'.format(args.tail_num)

if args.resolution == 128:
    src_dir = osp.join(args.dataset_path, 'silhouettes_cut128_pkl')
    des_dir = osp.join('{}_{}'.format(args.dataset_path, suffix), 'silhouettes_cut128_pkl_{}'.format(suffix))
elif args.resolution == 64:
    src_dir = osp.join(args.dataset_path, 'silhouettes_cut_pkl')
    des_dir = osp.join('{}_{}'.format(args.dataset_path, suffix), 'silhouettes_cut_pkl_{}'.format(suffix))   
print('src_dir={}'.format(src_dir))
print('des_dir={}'.format(des_dir))

if args.dataset.replace('-', '_') == 'CASIA_B':
    total_id_list = []
    for i in range(1, 125):
        id_name = '{:0>3d}'.format(i)
        if args.dataset.replace('-', '_') == 'CASIA_B' and id_name == '005':
            continue
        total_id_list.append(id_name)
    train_id_list = total_id_list[:73]
    test_id_list = total_id_list[73:]  
#############################################################
head_train_id_list = []
tail_train_id_list = []
tail_num = args.tail_num
head_num = len(train_id_list) - tail_num
if args.shuffle:
    np.random.seed(args.seed)
    np.random.shuffle(train_id_list)
head_train_id_list = sorted(train_id_list[:head_num])
tail_train_id_list = sorted(train_id_list[head_num:])
assert(len(head_train_id_list) + len(tail_train_id_list) == len(train_id_list))
#############################################################

print("####################################################")            
print('train_id_list={}, num={}'.format(train_id_list, len(train_id_list)))
print('test_id_list={}, num={}'.format(test_id_list, len(test_id_list)))
print('head_train_id_list={}, num={}'.format(head_train_id_list, len(head_train_id_list)))
print('tail_train_id_list={}, num={}'.format(tail_train_id_list, len(tail_train_id_list)))
print("####################################################")

def process_id(id0):
    id_path = os.path.join(src_dir, id0)
    if id0 in test_id_list or id0 in head_train_id_list:
        new_id = id0
        new_id_path = os.path.join(des_dir, new_id)
        os.makedirs(new_id_path, exist_ok=True)
        cmd = 'cp -r {}/* {}'.format(id_path, new_id_path)
        # print(cmd)
        os.system(cmd)
        print("Head ID={}".format(id0))
    elif id0 in tail_train_id_list:
        for type0 in sorted(os.listdir(id_path)):
            type_path = os.path.join(id_path, type0)
            if 'nm' in type0:
                new_id = id0
                new_id_path = os.path.join(des_dir, new_id)
                os.makedirs(new_id_path, exist_ok=True)
                cmd = 'cp -r {} {}'.format(type_path, new_id_path)
                # print(cmd)
                os.system(cmd)
        print("Tail ID={}".format(id0))
    else:
        print("####################################################")
        print("Ignore ID={}".format(id0))
        print("####################################################")              
    return

id_list = sorted(os.listdir(src_dir))
# for id0 in id_list:
#     process_id(id0)
from multiprocessing import Pool
pool = Pool()
pool.map(process_id, id_list)
pool.close()

# resplit after adding long-tailed data
new_head_train_id_list = head_train_id_list.copy()
new_tail_train_id_list = tail_train_id_list.copy()
new_train_id_list = (head_train_id_list + tail_train_id_list).copy()
new_test_id_list = test_id_list.copy()
del train_id_list, test_id_list, head_train_id_list, tail_train_id_list
assert(len(new_head_train_id_list) + len(new_tail_train_id_list) == len(new_train_id_list))
print("####################################################")
print('new_train_id_list={}, num={}'.format(new_train_id_list, len(new_train_id_list)))
print('new_test_id_list={}, num={}'.format(new_test_id_list, len(new_test_id_list)))
print('new_head_train_id_list={}, num={}'.format(new_head_train_id_list, len(new_head_train_id_list)))
print('new_tail_train_id_list={}, num={}'.format(new_tail_train_id_list, len(new_tail_train_id_list)))
print("####################################################")
pid_fname = osp.join('partition', '{}_{}.npy'.format(args.dataset, suffix))
if not osp.exists(pid_fname):
    pid_dict = {}
    pid_dict.update({'train_id_list':new_train_id_list})
    pid_dict.update({'test_id_list':new_test_id_list})
    pid_dict.update({'head_train_id_list':new_head_train_id_list})
    pid_dict.update({'tail_train_id_list':new_tail_train_id_list})
    np.save(pid_fname, pid_dict)
    print('{} Saved'.format(pid_fname))
else:
    pid_dict = np.load(pid_fname).item()
    assert(set(pid_dict['train_id_list']) == set(new_train_id_list))
    assert(set(pid_dict['test_id_list']) == set(new_test_id_list))
    assert(set(pid_dict['head_train_id_list']) == set(new_head_train_id_list))
    assert(set(pid_dict['tail_train_id_list']) == set(new_tail_train_id_list))
    print('{} Exists'.format(pid_fname))
