python test.py --config ./config/s3dis/WSegPC_cmg_rpc_seg.yaml \
    data_root /root/WSegPC_data/s3dis/s3dis_3d/ \
    data_root_img /root/WSegPC_data/s3dis/s3dis_2d/ \
    data_root_sp /root/WSegPC_data/s3dis/s3dis_3d/initial_superpoints_wypr \
    layers_2d 50 \
    test_batch_size 4 \
    test_workers 8 \
    view_num 6 \
    test_gpu [3] \
    split val \
    save_3d False \
    use_cls False \
    use_ema False \
    use_sp False \
    save_folder ./exp/s3dis/WSegPC_cmg_rpc_seg/ \
    save_path ./exp/s3dis/WSegPC_cmg_rpc_seg/ \
    model_path ./exp/s3dis/WSegPC_cmg_rpc_seg/model/model_best.pth.tar