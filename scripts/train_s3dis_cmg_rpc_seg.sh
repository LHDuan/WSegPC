python train_seg.py --config ./config/s3dis/WSegPC_cmg_rpc_seg.yaml \
    data_root /root/WSegPC_data/s3dis/s3dis_3d/ \
    data_root_img /root/WSegPC_data/s3dis/s3dis_2d/ \
    data_root_sp /root/WSegPC_data/s3dis/s3dis_3d/initial_superpoints_wypr/ \
    loop 16 \
    batch_size 4 \
    workers 8 \
    print_freq 100 \
    pseudo_label_3d True \
    pseudo_dir ./exp/s3dis/WSegPC_cmg_rpc/pseudo_label \
    pseudo_key pred_label \
    save_path ./exp/s3dis/WSegPC_cmg_rpc_seg/ \
    train_gpu [3]