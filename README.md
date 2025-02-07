

# CRF360D

Office source code of paper **Monocular 360 Depth Estimation via Spherical Fully-Connected CRFs**, [arXiv](https://arxiv.org/abs/2405.11564), [Project page](https://vlislab22.github.io/CRF360D/)



# Preparation

#### Installation

Environments


* python 3.9
* Pytorch 1.13.0, CUDA 11.7, torchvision 0.14.0
* Platform NVIDIA 3090


Install requirements

```bash
pip install -r requirements.txt
```

#### Datasets 

Please download the preferred datasets,  i.e., [Matterport3D](https://niessner.github.io/Matterport/), [Stanford2D3D](http://3dsemantics.stanford.edu/). For Matterport3D, please preprocess it following ([UniFuse/Matterport3D/README.md](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/tree/main/UniFuse)).



# Training 

#### CRF360D on Matterport3D

```
python train.py --config ./configs/train_matterport3d/b5_matterpot3d.yaml
```

#### CRF360D on Stanford2D3D

```
python train.py --config ./configs/train_stanford2d3d/b5_stanford2d3d.yaml
```

It is similar for other datasets, such as Structured3D dataset. 


# Evaluation  

#### Pre-trained models

The pre-trained models of CRF360D for 2 datasets are available, [Matterport3D](https://drive.google.com/drive/folders/1ewHLBTBKwCp37o2bnfNTlAeD5FaMAA-w?usp=sharing), and [Stanford2D3D](https://drive.google.com/drive/folders/1yX9OyL6GOuTTxaHSlqrB6RE93rEF8K9x?usp=sharing).

#### Test on a pre-trained model

```
python evaluate.py --config ./configs/test_matterport3d/panocrf_b5.yaml 
```



## Citation

Please cite our paper if you find our work useful in your research.

```
@article{cao2024crf360d,
  title={CRF360D: Monocular 360 Depth Estimation via Spherical Fully-Connected CRFs},
  author={Cao, Zidong and Wang, Lin},
  journal={arXiv preprint arXiv:2405.11564},
  year={2024}
}
```

