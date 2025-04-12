python train_seg.py --config ./config/scannet/WSegPC_cmg_rpc_seg.yaml \
    data_root /root/WSegPC_data/scannet/scannet_3d/ \
    data_root_img /root/WSegPC_data/scannet/scannet_2d/ \
    data_root_sp /root/WSegPC_data/scannet/scannet_3d/initial_superpoints_wypr \
    loop 4 \
    print_freq 100 \
    pseudo_label_3d True \
    pseudo_dir ./exp/scannet/WSegPC_cmg_rpc/pseudo_label \
    pseudo_key pred_label \
    save_path ./exp/scannet/WSegPC_cmg_rpc_seg/ \
    train_gpu [2]