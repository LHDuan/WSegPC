# WSegPC: High-quality Pseudo-labeling for Point Cloud Segmentation with Scene-level Annotation

This repository contains the official PyTorch implementation for the paper:

**High-quality Pseudo-labeling for Point Cloud Segmentation with Scene-level Annotation**

Our work addresses the challenging task of weakly supervised 3D point cloud semantic segmentation using only scene-level category tags. We propose a novel framework featuring:
- **Cross-Modal Guidance (CMG):** Utilizes point-wise contrastive distillation to align 3D point features with corresponding 2D pixel features from associated images, enhancing feature representation under weak supervision.
- **Region-Point Consistency (RPC):** Introduces a teacher-student framework with consistency regularization between point predictions and dynamically filtered regional semantics derived from unsupervised partitions, effectively refining noisy pseudo-labels.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/LHDuan/WSegPC.git
    cd WSegPC
    ```

2.  **Create Conda Environment:** We recommend using Anaconda or Miniconda to manage dependencies.
    ```bash
    conda create -n wsegpc python=3.8 -y
    conda activate wsegpc
    ```

3.  **Install Dependencies:** Our codebase builds upon [Point Transformer V2](https://github.com/Pointcept/PointTransformerV2/). Please follow their installation instructions for core dependencies like PyTorch and CUDA. Our development environment used the following versions:
    - Python: `3.8.15`
    - PyTorch: `1.10.1`
    - CUDA Toolkit: `11.1`
    - [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine): `0.5.4` (Installation might require specific compilation steps, please refer to their documentation)



## Data Preparation
Please download our processed ScanNet v2 and S3DIS datasets from [here](https://whueducn-my.sharepoint.com/:f:/g/personal/2014302590165_whu_edu_cn/EokVow3JimhKqs8XwO7Qa0MBU-F8YfJe-KlRdoMgVmAiRg?e=FLzbpv) and link the processed dataset directory to the codebase.

The structure of the data folder should be:

    ```
    /path/to/your/data/root/ # e.g., /root/WSegPC_data/
      ├── scannet/
      │   ├── scannet_3d/
      │   │   ├── train/          # Contains train .pth files
      │   │   ├── val/            # Contains val .pth files
      │   │   └── initial_superpoints_wypr/ # Contains WyPR partitions
      │   └── scannet_2d/
      │       ├── scene0000_00/   # Example scene
      │       └── ...
      └── s3dis/
          ├── s3dis_3d/
          │   ├── Area_1/         # Area 1 .npz files
          │   ├── .../            # Area 2-6 .npz files
          │   └── initial_superpoints_wypr/ # Contains WyPR partitions
          └── s3dis_2d/
              ├── Area_1/         # Multi-view images for Area 1
              ├── Area_2/         # Multi-view images for Area 2
              └── ...             # Etc. for other areas
    ```

## Training Pipeline

Our method involves a multi-stage training process. The following scripts provide examples for running each stage on the ScanNet dataset. Adapt them for S3DIS as needed.

**Stage 1: Train with Cross-Modal Guidance (CMG)**

Trains the initial model focusing on aligning 2D and 3D features using scene-level labels and the CMG loss.

```bash
# Example for ScanNet (modify paths and GPU IDs as needed)
sh scripts/train_scannet_cmg.sh
```
* Checkpoints will be saved under `./exp/scannet/WSegPC_cmg/`.

**Stage 2: Train with CMG + Region-Point Consistency (RPC)**

Initializes with Stage 1 weights and adds the RPC loss for refinement.

```bash
# Example for ScanNet (modify paths and GPU IDs as needed)
sh scripts/train_scannet_cmg_rpc.sh
```
* This stage trains the model responsible for generating the final high-quality pseudo-labels. Checkpoints saved under `./exp/scannet/WSegPC_cmg_rpc/`.

**Stage 3: Train Segmentation Network with Pseudo-Labels**

Trains a standard segmentation network using the pseudo-labels generated in Stage 2.

```bash
# Example for ScanNet (modify paths and GPU IDs as needed)
sh scripts/train_scannet_cmg_rpc_seg.sh
```
* Ensure the `pseudo_dir` points to the location where pseudo-labels are saved (typically generated during evaluation of the Stage 2 model on the training set).

## Evaluation
Download the example pre-trained models from [here](https://whueducn-my.sharepoint.com/:f:/g/personal/2014302590165_whu_edu_cn/EokVow3JimhKqs8XwO7Qa0MBU-F8YfJe-KlRdoMgVmAiRg?e=FLzbpv) and place them under the corresponding `exp/` subdirectories (e.g., `exp/scannet/WSegPC_cmg_rpc/model/`).

1. **Generate Pseudo-Labels & Evaluate Quality:**

Use the Stage 2 model (CMG+RPC) to generate pseudo-labels for the training set and evaluate their mIoU against ground truth 
```bash
# Example for ScanNet (modify paths and GPU IDs as needed)
sh scripts/test_scannet_wseg.sh
```

2. **Evaluate Final Segmentation Performance:**

Evaluate the performance of the final segmentation model (trained in Stage 3) on the validation set.
```bash
# Example for ScanNet (modify paths and GPU IDs as needed)
sh scripts/test_scannet_seg.sh
```

## Citations
If you find our work useful in your research, please consider citing:
```
@inproceedings{duan2023wsegpc,
  title={High-quality Pseudo-labeling for Point Cloud Segmentation with Scene-level Annotation},
  author={Duan, Lunhao and Zhao, Shanshan and Weng, Xingxing and Zhang, Jing and Xia, Gui-Song},
  booktitle={},
  year={2025}
}
```

## Contact
[lhduan@whu.edu.cn](lhduan@whu.edu.cn)

## Acknowledgements
This implementation heavily builds upon the excellent codebases of:
- [BPNet](https://github.com/wbhu/BPNet)
- [Point Transformer V2](https://github.com/Pointcept/PointTransformerV2/)
- [OpenScene](https://github.com/pengsongyou/openscene)
- [WyPR](https://github.com/facebookresearch/WyPR)
- [GrowSP](https://github.com/vLAR-group/GrowSP)

We sincerely thank the authors for making their code publicly available.