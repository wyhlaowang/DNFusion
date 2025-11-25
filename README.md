# DNFusion

Code for DNFuison: Infrared-Visible Image Fusion through Feature-Based Decomposition and Domain Normalization

[Weiyi Chen](https://github.com/wyhlaowang/DNFusion), 
[Lingjuan Miao](https://github.com/wyhlaowang/DNFusion), 
[Yuhao Wang](https://github.com/wyhlaowang/DNFusion)
[Zhiqiang Zhou](https://github.com/bitzhouzq) and 
[Yajun Qiao](https://github.com/QYJ123/)

# Usage
**1. Create Environment**
```
# install cuda
Recommended cuda11.1

# create conda environment
conda create -n LDFusion python=3.9.12
conda activate LDFusion

# select pytorch version yourself (recommended torch 1.8.2)
pip install -r requirements.txt
```

**2. Data Preparation, training and inference**

You can put your own test data directly into the ```test_imgs/rs``` directory, and run ```python test.py``` at directory of ```./src/```.

Then, the fused results will be saved in the ```./self_results/rs/``` folder.

If you want to train the model with your own data, change the value of ```data_dir``` in the ```m3fd.py``` file to the address of your dataset.

We recommend inference and training in cuda because DCN libraries do not run well in cpu.

# DNFusion
<img src="docs/overview.png" width="1000">

Our proposed UNIFusion is an autoencoder structure, which consists of image decomposition, feature extraction, fusion, and reconstruction modules. The feature extraction module is a three-branch network based on dense attention, consisting of encoders E_ir, E_vi, and E_u_, which are used to extract unique and unified features. 

<img src="docs/ufs.png" width="1000">

The fusion and reconstruction module is devised to fuse features and generate fusion results, while employing a non-local Gaussian filter to reduce the adverse impact of noise on the fusion quality. 

<img src="docs/dec.png" width="600">

Specifically, we decompose infrared--visible images into common regions (C_vi and C_ir) and unique regions (P_vi and P_ir). The dense attention is leveraged to effectively extract features from the common and unique regions. 

<img src="docs/nlgs.png" width="500">

To eliminate modal differences, we propose the unified feature space to transform infrared features into the pseudo-visible domain. As noisy source images may degrade the fusion quality, we design a non-local Gaussian filter to minimize the impact of noise on the fusion results while maintaining the image details.


# Fusion Results
<img src="docs/fu1.png" width="800">

<img src="docs/fu2.png" width="800">

# Abstract
Infrared-visible image fusion is valuable across various applications due to the complementary information that it provides. However, the current fusion methods face challenges in achieving high-quality fused images. This paper identifies a limitation in the existing fusion framework that affects the fusion quality: modal differences between infrared and visible images are often overlooked, resulting in the poor fusion of the two modalities. This limitation implies that features from different sources may not be consistently fused, which can impact the quality of the fusion results. Therefore, we propose a framework that utilizes feature-based decomposition and domain normalization. This decomposition method separates infrared and visible images into common and unique regions. To reduce modal differences while retaining unique information from the source images, we apply domain normalization to the common regions within the unified feature space. This space can transform infrared features into a pseudo-visible domain, ensuring that all features are fused within the same domain and minimizing the impact of modal differences during the fusion process. Noise in the source images adversely affects the fused images, compromising the overall fusion performance. Thus, we propose the non-local Gaussian filter. This filter can learn the shape and parameters of its filtering kernel based on the image features, effectively removing noise while preserving details. Additionally, we propose a novel dense attention in the feature extraction module, enabling the network to understand and leverage inter-layer information. Our experiments demonstrate a marked improvement in fusion quality with our proposed method.


If this work is helpful to you, please cite it as:

```
@Article{rs16060969,
AUTHOR = {Chen, Weiyi and Miao, Lingjuan and Wang, Yuhao and Zhou, Zhiqiang and Qiao, Yajun},
TITLE = {Infrared–Visible Image Fusion through Feature-Based Decomposition and Domain Normalization},
JOURNAL = {Remote Sensing},
VOLUME = {16},
YEAR = {2024},
NUMBER = {6},
ARTICLE-NUMBER = {969},
URL = {https://www.mdpi.com/2072-4292/16/6/969},
ISSN = {2072-4292},
DOI = {10.3390/rs16060969}
}
```
