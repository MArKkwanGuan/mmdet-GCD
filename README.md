#  Gaussian Combined Distance: A Generic Metric for Object Detection

This is the official code for the [GCD]. The method is accepted by the GRSL in 2025.

## Installation

### Requirements

- Linux
- Python 3.7 (Python 2 is not supported)
- PyTorch **1.5** or higher
- CUDA 10.1 or higher
- NCCL 2
- GCC(G++) **5.4** or higher
- [mmcv-nwd](https://github.com/jwwangchn/mmcv-nwd.git)==**1.3.5**
- [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)==**12.0.3**

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n gcd python=3.7 -y
conda activate gcd
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

c. Install MMCV-NWD

```shell
git clone https://github.com/jwwangchn/mmcv-nwd.git
cd mmcv-nwd
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ../
```

d. Install COCOAPI-AITOD for Evaluating on AI-TOD dataset
```shell
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

e. Install

```shell

# optional
pip install -r requirements.txt

python setup.py develop
# or "pip install -v -e ."
```

## Prepare datasets

Please refer to [AI-TOD](https://github.com/jwwangchn/AI-TOD) for AI-TOD dataset.

## Run

The GCD's config files are in [configs/gcd].

Please see MMDetection full tutorials [with existing dataset](docs/1_exist_data_model.md) for beginners.

### Training on a single GPU

```shell
python tools/train.py configs/gcd/retinanet_r50_aitodv2_gcd_1x.py
```

## Benchmark

The benchmark and trained models will be publicly available soon.

## Citation
```BibTeX
@inproceedings{AI-TOD_2020_ICPR,
    title={Tiny Object Detection in Aerial Images},
    author={Wang, Jinwang and Yang, Wen and Guo, Haowen and Zhang, Ruixiang and Xia, Gui-Song},
    booktitle=ICPR,
    pages={3791--3798},
    year={2021},
}
```
```BibTeX
@ARTICLE{10847878,
  author={Guan, Ziqian and Fu, Xieyi and Huang, Pengjun and Zhang, Hengyuan and Du, Hubin and Liu, Yongtao and Wang, Yinglin and Ma, Qang},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Gaussian Combined Distance: A Generic Metric for Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Measurement;Object detection;Feature extraction;Optimization;Detectors;Geoscience and remote sensing;Accuracy;Training;Sensitivity;Convergence;Tiny Object Detection;Generic Metric},
  doi={10.1109/LGRS.2025.3531970}}
```
