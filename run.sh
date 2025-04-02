# Cloth-Imbalanced Gait Recognition via Hallucination

# IM-Refer
export MODEL=AMP_DDP_casia_b_tn00_noshuffle_rt128_train_base_bin16 && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
--dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_tn00_noshuffle/silhouettes_cut128_pkl_tn00_noshuffle \
--pid_fname partition/CASIA_B_tn00_noshuffle.npy --tail_split False \
--head_batch_size 8 16 --tail_batch_size 0 16 \
--milestones 10000 20000 30000 --total_iter 35000 --warmup False \
--more_channels False --bin_num 16 --hidden_dim 256 \
--encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 \
--model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
--AMP True --DDP True \
2>&1 | tee $MODEL.log

# IM-Basic
export MODEL=AMP_DDP_casia_b_tn53_noshuffle_rt128_train_base_bin16 && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
--dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_tn53_noshuffle/silhouettes_cut128_pkl_tn53_noshuffle \
--pid_fname partition/CASIA_B_tn53_noshuffle.npy --tail_split False \
--head_batch_size 8 16 --tail_batch_size 0 16 \
--milestones 10000 20000 30000 --total_iter 35000 --warmup False \
--more_channels False --bin_num 16 --hidden_dim 256 \
--encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 \
--model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
--AMP True --DDP True \
2>&1 | tee $MODEL.log

# CCH-Basic
export MODEL=AMP_DDP_casia_b_tn53_noshuffle_rt128_train_base_bin16_hbs4x16_tbs4x16 && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
--dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_tn53_noshuffle/silhouettes_cut128_pkl_tn53_noshuffle \
--pid_fname partition/CASIA_B_tn53_noshuffle.npy --tail_split True \
--head_batch_size 4 16 --tail_batch_size 4 16 \
--milestones 10000 20000 30000 --total_iter 35000 --warmup False \
--more_channels False --bin_num 16 --hidden_dim 256 \
--encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 \
--model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
--AMP True --DDP True \
2>&1 | tee $MODEL.log

find . -name *35000*ptm | xargs -i cp {} .

# CCH
export MODEL=AMP_DDP_casia_b_tn53_noshuffle_rt128_train_base_bin16_hbs4x16_tbs4x16_JTCCV23 && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
--dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_tn53_noshuffle/silhouettes_cut128_pkl_tn53_noshuffle \
--pid_fname partition/CASIA_B_tn53_noshuffle.npy --tail_split True \
--head_batch_size 4 16 --tail_batch_size 4 16 \
--milestones 10000 20000 30000 --total_iter 35000 --warmup False \
--more_channels False --bin_num 16 --hidden_dim 256 \
--encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 \
--model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
--AMP True --DDP True \
--cc_aug True --cc_offupdate_interval 1000 --cc_onupdate_interval 1 --cc_onupdate_momentum 0.99 \
--lr 0.01 --cc_lr 0.1 --cc_k 3 --cc_s 2.0 --cc_normalize False --clip_grad_norm 5.0 \
--cc_triplet_weight 1.0 --cc_triplet_margin 0.2 \
--init_model AMP_DDP_casia_b_tn53_noshuffle_rt128_train_base_bin16_hbs4x16_tbs4x16_CASIA-B_73_False-35000-encoder.ptm \
2>&1 | tee $MODEL.log

# Eval IM-Refer/IM-Basic/CCH-Basic
python -u test.py \
--dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_tn00_noshuffle/silhouettes_cut128_pkl_tn00_noshuffle \
--pid_fname partition/CASIA_B_tn00_noshuffle.npy --tail_split False \
--more_channels False --bin_num 16 --hidden_dim 256 \
--head_batch_size 1 --gpu 0,1,2,3 --test_set test \
--resume False --ckp_prefix 

# Eval CCH
python -u test.py \
--dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_tn53_noshuffle/silhouettes_cut128_pkl_tn53_noshuffle \
--pid_fname partition/CASIA_B_tn53_noshuffle.npy --tail_split True \
--more_channels False --bin_num 16 --hidden_dim 256 \
--head_batch_size 1 --gpu 0,1,2,3 --test_set test \
--cc_aug True --feat_idx 1 --resume False --ckp_prefix