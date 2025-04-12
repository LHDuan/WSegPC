python train_WSegPC.py --config ./config/scannet/WSegPC_cmg.yaml \
    data_root /root/WSegPC_data/scannet/scannet_3d/ \
    data_root_img /root/WSegPC_data/scannet/scannet_2d/ \
    data_root_sp /root/WSegPC_data/scannet/scannet_3d/initial_superpoints_wypr \
    layers_2d 50 \
    view_num 6 \
    loop 4 \
    print_freq 100 \
    save_path ./exp/scannet/WSegPC_cmg/ \
    train_gpu [1]
