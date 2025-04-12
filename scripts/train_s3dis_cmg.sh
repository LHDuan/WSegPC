python train_WSegPC.py --config ./config/s3dis/WSegPC_cmg.yaml \
    data_root /root/WSegPC_data/s3dis/s3dis_3d/ \
    data_root_img /root/WSegPC_data/s3dis/s3dis_2d/ \
    data_root_sp /root/WSegPC_data/s3dis/s3dis_3d/initial_superpoints_wypr/ \
    layers_2d 50 \
    view_num 6 \
    loop 16 \
    print_freq 100 \
    save_path ./exp/s3dis/WSegPC_cmg/ \
    train_gpu [0]