# Super-resolution (In progress...)

这个仓库主要是为了个人学习使用，很多资源来自于微信公众号的推荐、[arXiv](https://arxiv.org/) 网站、重要的国际会议（CVPR、ECCV、ICCV、NeuraIPS、ICLR）、GitHub 等。

This repository is mainly for personal study. Most of resources come from the WeChat recommendation, [arXiv](https://arxiv.org/) website, important conferences (CVPR, ECCV, ICCV, NeuraIPS and ICLR), GitHub, etc.

## Overview

> - [Important Repositories](#Important-Repositories)
>   - [Awesome SR Lists](#Awesome-SR-Lists)
>   - [Awesome Repositories](#Awesome-Repositories)
>   - [SR Metrics](#SR-Metrics)
> - [Datasets](#Datasets)
> - [Papers](#Papers)
>   - [Non-DL Based Approach](#Non-DL-Based-Approach)
>   - [DL Based Approach](#DL-Based-Approach)
>   - [Super Resolution Survey](#Super-Resolution-Survey) 
> - [Workshops](#Workshops)
> - [Excellent  Personal Website](#Excellent-Personal-Website)

## Important Repositories

### Awesome SR Lists

[Single-Image-Super-Resolution](https://github.com/YapengTian/Single-Image-Super-Resolution)

[Super-Resolution.Benckmark](https://github.com/huangzehao/Super-Resolution.Benckmark)

[Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution)

[VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution)

[Awesome Super-Resolution](https://github.com/ptkin/Awesome-Super-Resolution)

[Awesome-LF-Image-SR](https://github.com/YingqianWang/Awesome-LF-Image-SR)

[Awesome-Stereo-Image-SR](https://github.com/YingqianWang/Awesome-Stereo-Image-SR)

[AI-video-enhance](https://github.com/jlygit/AI-video-enhance)

### Awesome Repositories

|                         Repositories                         | Framework |      |                         Repositories                         | Framework  |
| :----------------------------------------------------------: | :-------: | ---- | :----------------------------------------------------------: | :--------: |
| [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)  |  PyTorch  |      | [Image-Super-Resolution](https://github.com/titu1994/Image-Super-Resolution) |   Keras    |
|      [RCAN-PyTorch](https://github.com/yulunzhang/RCAN)      |  PyTorch  |      | [image-super-resolution](https://github.com/idealo/image-super-resolution) |   Keras    |
|   [CARN-PyTorch](https://github.com/nmhkahn/CARN-pytorch)    |  PyTorch  |      | [super-resolution](https://github.com/krasserm/super-resolution) |   Keras    |
|        [BasicSR](https://github.com/xinntao/BasicSR)         |  PyTorch  |      | [Super-Resolution-Zoo](https://github.com/WolframRhodium/Super-Resolution-Zoo) |   MxNet    |
| [Super-resolution](https://github.com/icpm/super-resolution) |  PyTorch  |      |  [neural-enhance](https://github.com/alexjc/neural-enhance)  |   Theano   |
| [Video-super-resolution](https://github.com/thangvubk/video-super-resolution) |  PyTorch  |      |          [srez](https://github.com/david-gpu/srez)           | Tensorflow |
|          [MMSR](https://github.com/open-mmlab/mmsr)          |  PyTorch  |      |        [waifu2x](https://github.com/nagadomi/waifu2x)        |   Torch    |

### SR Metrics

Note this table is referenced from [here](https://github.com/ptkin/Awesome-Super-Resolution).

| Metric  | Papers                                                       |
| :------ | ------------------------------------------------------------ |
| MS-SSIM | **Multiscale structural similarity for image quality assessment**, *Wang, Zhou; Simoncelli, Eero P.; Bovik, Alan C.*, **ACSSC 2003**, [[ACSSC](https://ieeexplore.ieee.org/document/1292216)], `MS-SSIM` |
| SSIM    | **Image Quality Assessment: From Error Visibility to Structural Similarity**, *Wang, Zhou; Bovik, Alan C.; Sheikh, Hamid R.; Simoncelli, Eero P*, **TIP 2004**, [[TIP](https://ieeexplore.ieee.org/document/1284395)], `SSIM` |
| IFC     | **An information fidelity criterion for image quality assessment using natural scene statistics**, *Sheikh, Hamid Rahim; Bovik, Alan Conrad; de Veciana, Gustavo de Veciana*, **TIP 2005**, [[TIP](https://ieeexplore.ieee.org/document/1532311/)], `IFC` |
| VIF     | **Image information and visual quality**, *Sheikh, Hamid Rahim; Bovik, Alan C.*, **TIP 2006**, [[TIP](https://ieeexplore.ieee.org/document/1576816)], `VIF` |
| FSIM    | **FSIM: A Feature Similarity Index for Image Quality Assessment**, *Zhang, Lin; Zhang, Lei; Mou, Xuanqin; Zhang, David*, **TIP 2011**, [[Project](http://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm)], [[TIP](https://ieeexplore.ieee.org/document/5705575)], `FSIM` |
| NIQE    | **Making a “Completely Blind” Image Quality Analyzer**, *Mittal, Anish; Soundararajan, Rajiv; Bovik, Alan C.*, **Signal Processing Letters 2013**, [[Matlab*](https://github.com/csjunxu/Bovik_NIQE_SPL2013)], [[Signal Processing Letters](https://ieeexplore.ieee.org/document/6353522)], `NIQE` |
| Ma      | **Learning a no-reference quality metric for single-image super-resolution**, *Ma, Chao; Yang, Chih-Yuan; Yang, Xiaokang; Yang, Ming-Hsuan*, **CVIU 2017**, [[arXiv](https://arxiv.org/abs/1612.05890)], [[CVIU](https://www.sciencedirect.com/science/article/pii/S107731421630203X)], [[Matlab*](https://github.com/chaoma99/sr-metric)], [[Project](https://sites.google.com/site/chaoma99/sr-metric)], `Ma` |

## Datasets

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution#link-of-datasets).

| Name                    |   Usage    |                             Link                             |           Comments            |
| :---------------------- | :--------: | :----------------------------------------------------------: | :---------------------------: |
| Set5                    |    Test    | [download](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip) | super-resolution test dataset |
| SET14                   |    Test    | [download](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip) | super-resolution test dataset |
| BSD100                  |    Test    | [download](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip) | super-resolution test dataset |
| Urban100                |    Test    | [download](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip) | super-resolution test dataset |
| Manga109                |    Test    | [download](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip) | super-resolution test dataset |
| BSD300                  | Train/Val  | [download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz) |                               |
| BSD500                  | Train/Val  | [download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) |                               |
| 91-Image                |   Train    | [download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar) |             Yang              |
| DIV2K2017               | Train/Val  |     [website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)     |           NTIRE2017           |
| Flickr2K                |   Train    |  [download](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)  |                               |
| Real SR                 | Train/Val  | [website](https://competitions.codalab.org/competitions/21439#participate) |           NTIRE2019           |
| Waterloo                |   Train    |   [website](https://ece.uwaterloo.ca/~k29ma/exploration/)    |                               |
| VID4                    |    Test    | [download](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip) |           4 videos            |
| MCL-V                   |   Train    |        [website](http://mcl.usc.edu/mcl-v-database/)         |           12 videos           |
| GOPRO                   | Train/Val  | [website](https://github.com/SeungjunNah/DeepDeblur_release) |       33 videos, deblur       |
| CelebA                  |   Train    | [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  |          Human faces          |
| Sintel                  | Train/Val  |       [website](http://sintel.is.tue.mpg.de/downloads)       |         Optical flow          |
| FlyingChairs            |   Train    | [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) |         Optical flow          |
| Vimeo-90k               | Train/Test |           [website](http://toflow.csail.mit.edu/)            |         90k HQ videos         |
| SR-RAW                  | Train/Test | [website](https://ceciliavision.github.io/project-pages/project-zoom.html) |   raw sensor image dataset    |
| Benchmark and DIV2K(SR) | Train/Test | [website](https://drive.google.com/drive/folders/1-99XFJs_fvQ2wFdxXrnJFcRRyPJYKN0K) |   super-resolution dataset    |

## Papers

### Non-DL Based Approach

| SCSR: TIP2010, Jianchao Yang et al.      | [paper](https://ieeexplore.ieee.org/document/5466111/?arnumber=5466111) | [code](http://www.ifp.illinois.edu/~jyang29/)                |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ANR: ICCV2013, Radu Timofte et al.       | [paper](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-ICCV-2013.pdf) | [code](http://www.vision.ee.ethz.ch/~timofter/ICCV2013_ID1774_SUPPLEMENTARY/index.html) |
| A+: ACCV 2014, Radu Timofte et al.       | [paper](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-ACCV-2014.pdf) | [code](http://www.vision.ee.ethz.ch/~timofter/ACCV2014_ID820_SUPPLEMENTARY/) |
| IA: CVPR2016, Radu Timofte et al.        | [paper](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPR-2016.pdf) |                                                              |
| SelfExSR: CVPR2015, Jia-Bin Huang et al. | [paper](https://uofi.box.com/shared/static/8llt4ijgc39n3t7ftllx7fpaaqi3yau0.pdf) | [code](https://github.com/jbhuang0604/SelfExSR)              |
| NBSRF: ICCV2015, Jordi Salvador et al.   | [paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Salvador_Naive_Bayes_Super-Resolution_ICCV_2015_paper.pdf) |                                                              |
| RFL: ICCV2015, Samuel Schulter et al     | [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schulter_Fast_and_Accurate_2015_CVPR_paper.pdf) | [code](<https://www.tugraz.at/institute/icg/research/team-bischof/samuel-schulter/>) |

### DL Based Approach

|                   2014-2017                    |                 2018                 |                 2019                 |                 2020                 |
| :--------------------------------------------: | :----------------------------------: | :----------------------------------: | :----------------------------------: |
| [**Papers 2014-2017.md**](Papers-2014-2017.md) | [**Papers 2018.md**](Papers-2018.md) | [**Papers 2019.md**](Papers-2019.md) | [**Papers 2020.md**](Papers-2020.md) |

### Survey

[1] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, Jing-Hao Xue. Deep Learning for Single Image Super-Resolution: A Brief Review. arxiv, 2018. [paper](https://arxiv.org/pdf/1808.03344.pdf)

[2] Saeed Anwar, Salman Khan, Nick Barnes. A Deep Journey into Super-resolution: A survey. arxiv, 2019.[paper](https://arxiv.org/pdf/1904.07523.pdf)

[3] Wang, Z., Chen, J., & Hoi, S. C. (2019). Deep learning for image super-resolution: A survey. arXiv preprint arXiv:1902.06068.[paper](https://arxiv.org/abs/1902.06068)

[4] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, Jing-Hao Xue (2019). Deep Learning for Single Image Super-Resolution: A Brief Review [paper](https://arxiv.org/pdf/1808.03344.pdf)

[5] Hongying Liu,Zhubo Ruan.(2020). Video Super Resolution Based on Deep Learning: A comprehensive survey.[paper](https://arxiv.org/abs/2007.12928)



## Workshops 

| [NTIRE17](ttp://openaccess.thecvf.com/CVPR2017_workshops/CVPR2017_W12.py) | [NTIRE18](http://openaccess.thecvf.com/CVPR2018_workshops/CVPR2018_W13.py) | [PIRM18](https://pirm2018.org/) | [NTIRE19](http://openaccess.thecvf.com/CVPR2019_workshops/CVPR2019_NTIRE.py) | [AIM19](https://openaccess.thecvf.com/ICCV2019_workshops/ICCV2019_AIM) | [NTIRE20](http://openaccess.thecvf.com/CVPR2020_workshops/CVPR2020_w31.py) | [AIM20](https://data.vision.ee.ethz.ch/cvl/aim20/) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
|                                                              |                                                              |                                 |                                                              |                                                              |                                                              |                                                    |



## ExcellentPersonal Website

| [Manri Cheon](https://manricheon.github.io/) | [Yulun Zhang](http://yulunzhang.com/) | [Yapeng Tian](http://yapengtian.org/) | [Xintao Wang](https://xinntao.github.io/) |
| -------------------------------------------- | ------------------------------------- | ------------------------------------- | ----------------------------------------- |
|                                              |                                       |                                       |                                           |









