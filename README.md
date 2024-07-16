# Eliminating Feature Ambiguity for Few-Shot Segmentation

This repository contains the code for our ECCV 2024 [paper](https://arxiv.org/abs/2407.09842) "*Eliminating Feature Ambiguity for Few-Shot Segmentation*", where we design a plug-in network AENet for three existing cross attention-based baselines: [CyCTR](https://github.com/YanFangCS/CyCTR-Pytorch), [SCCAN](https://github.com/Sam1224/SCCAN) and [HDMNet](https://github.com/Pbihao/HDMNet).

## Dependencies

Please follow the official guidelines of the selected baselines to create their virtual environments. Note that:
- You can follow [SCCAN](https://github.com/Sam1224/SCCAN) to create a shared environment for both CyCTR and SCCAN, but **remember to run the following commands to install Deformable Attention for CyCTR**:
  ```
  > cd SCCAN_CyCTR/model/ops/
  > bash make.sh
  ```
- You need to follow [HDMNet](https://github.com/Pbihao/HDMNet) to create a separate environment for HDMNet.

## Directory Structure

The directory structure is:

    ../
    ├── SCCAN_CyCTR/  # code for SCCAN and CyCTR
    ├── HDMNet/       # code for HDMNet
    ├── lists/        # shared data lists
    ├── initmodel/    # shared pretrained backbones
    └── data/         # shared data
        ├── base_annotation/
        ├── VOCdevkit2012/
        │   └── VOC2012/
        └── MSCOCO2014/           
            ├── annotations/
            ├── train2014/
            └── val2014/

### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

You can download the pre-processed PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> datasets [here](https://entuedu-my.sharepoint.com/:f:/g/personal/qianxion001_e_ntu_edu_sg/ErEg1GJF6ldCt1vh00MLYYwBapLiCIbd-VgbPAgCjBb_TQ?e=ibJ4DM), and extract them into `data/` folder. Then, you need to create a symbolic link to the `pascal/VOCdevkit` data folder as follows:
```
> ln -s <absolute_path>/data/pascal/VOCdevkit <absolute_path>/data/VOCdevkit2012
```

### Data Lists

- Download the data lists from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/Eateth41QipCrBTv3e8bkKYBzj2jfLO6u9ShC55l0ARLtA?e=8uFgtC) and put them into the `lists/` directory.
- Three baselines share the same data lists, you can put the `lists/` folder in the root directory, and create symbolic links as follows:
```
> ln -s <absolute_path>/lists <absolute_path>/SCCAN_CyCTR/lists
> ln -s <absolute_path>/lists <absolute_path>/HDMNet/lists
```

### Backbones

- Download the pretrained backbones from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EaAPysvXJbZGuCpZqN1BtEkBOiAeuE6cln6hD2hGJOvgKA?e=kzZHLf) and put them into the `initmodel/` directory.
- Three baselines share the same pretrained backbones, you can put the `initmodel/` in the root directory, and create symbolic links as follows:
```
> ln -s <absolute_path>/initmodel <absolute_path>/SCCAN_CyCTR/initmodel
> ln -s <absolute_path>/initmodel <absolute_path>/HDMNet/initmodel
```

## Pretrained Models

- Download [exp_cyctr.zip](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EUrFo0U_0n9KhUYagPNmPo8BYmt68hNDYCeY8f6I5g1igQ?e=JZaHJD) to obtain CyCTR-related pretrained models, and extract them to `SCCAN_CyCTR/exp`.
- Download [exp_sccan.zip](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/Eeoyp3enpKRFiiAxxlyq-bsBatvlFDZx6eMLzvHgNjiG1g?e=evIuEs) to obtain SCCAN-related pretrained models, and extract them to `SCCAN_CyCTR/exp`.
- Download [exp_hdmnet.zip](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/ERnKxHBd9o1NiG_QvWKix9gBIcELffIrIOTaHGP9lEYYSg?e=jMoBxe) to obtain HDMNet-related pretrained models, and extract them to `HDMNet/exp`.

## Training and Testing Commands

- **Training (with 4 GPUs)**
  - Take SCCANPlus as an example:
    ```
    > cd SCCAN_CyCTR
    
    # ============================================================
    # Args for train.sh:
    #   exp_name: split0/split1/split2/split3 - different folds
    #   dataset:  pascal/coco - two benchmark datasets
    #   port:     1234 - port for DDP
    #   arch:     SCCANPlus/CyCTRPlus - model names
    #   net:      vgg/resnet50 - pretrained backbones
    #   postfix:  ddp/ddp_5s - 1 or 5 shot
    # ============================================================
    > sh train.sh split0 pascal 1234 SCCANPlus resnet50 ddp
    ```
- **Testing (with 1 GPUs)**
  - Take SCCANPlus as an example:
    ```
    > cd SCCAN_CyCTR
    > sh test.sh split0 pascal SCCANPlus resnet50 ddp
    ```

## References

This repo is mainly built based on [BAM](https://github.com/chunbolang/BAM) and [SCCAN](https://github.com/sam1224/SCCAN). Thanks for their great work!

