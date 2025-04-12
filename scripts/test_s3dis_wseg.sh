python test.py --config ./config/s3dis/WSegPC_cmg_rpc.yaml \
    data_root /root/WSegPC_data/s3dis/s3dis_3d/ \
    data_root_img /root/WSegPC_data/s3dis/s3dis_2d/ \
    data_root_sp /root/WSegPC_data/s3dis/s3dis_3d/initial_superpoints_wypr/ \
    layers_2d 50 \
    test_batch_size 4 \
    test_workers 8 \
    view_num 6 \
    test_gpu [3] \
    split train \
    save_3d True \
    use_cls True \
    use_ema True \
    use_sp True \
    save_folder ./exp/s3dis/WSegPC_cmg_rpc/pseudo_label \
    save_path ./exp/s3dis/WSegPC_cmg_rpc/ \
    model_path ./exp/s3dis/WSegPC_cmg_rpc/model/model_best.pth.tar
