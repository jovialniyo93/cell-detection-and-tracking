
# [A Weakly Supervised Learning Method for Cell Detection and Tracking Using Incomplete Initial Annotations](https://www.mdpi.com/1422-0067/24/22/16028/pdf)

The code in this repository is supplementary to our publication **Weakly Supervised Learning Method for Cell Detection and Tracking Using Incomplete Initial Annotations** 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB
* This project is writen in Python 3 and makes use of Pytorch. 

## Installation
In order to get the code, either clone the project, or download a zip file from GitHub.

Clone the Cell detection and Tracking repository:
```
https://github.com/jovialniyo93/cell-detection-and-tracking.git
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the Cell detection and Tracking repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
```
```
conda env create -f requirements.yml
```
Activate the virtual environment cell_detection_and_tracking_ve:
```
conda activate cell_detection_and_tracking_ve
```

# How to train and test our model

```ips_cell``` folder contains all scripts used to train and test models. More details on how to train and test the code is mentioned inside ```ips_cell``` folder.

<br/>

## Independent dataset

```fluo_gowt``` folder contains all scripts used to train and test the independent dataset.

In this section, it is described how to reproduce the detection and tracking results on public dataset using our method. Download the Cell Tracking Challenge training data sets [Fluo-N2DH-GOWT1](http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip) Unzip the data sets into the folder *fluo_gowt*. Download the [evaluation software](http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip) from the Cell Tracking Challenge and unzip it in the repository. The procedure followed to train and test the independent dataset is the same as the one used for ```ips_cell```


# Project Collaborators and Contact

**Authors:** Wu Hao, Jovial Niyogisubizo, Zhao Keliang, Jintao Meng, Wenhui Xi, Hongchang Li, Yanjie Wei

Copyright & copy, 2023. Wu Hao, Jovial Niyogisubizo, Zhao Keliang, Jintao Meng, Wenhui Xi, Hongchang Li, Yanjie Wei and Curators of the Center for High Performance Computing, Shenzhen Institute of Advanced Technology, CAS. All Rights Reserved.

**Created by:** Ph.D. student: Jovial Niyogisubizo 
Department of Computer Applied Technology,  
Center for High Performance Computing, Shenzhen Institute of Advanced Technology, CAS. 

For more information, contact:

* **Prof Yanjie Wei**  
Shenzhen Institute of Advanced Technology, CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
yj.wei@siat.ac.cn


* **Jovial Niyogisubizo**  
Shenzhen Institute of Advanced Tech., CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
jovialniyo93@gmail.com

**Citation Request:** 

If you find our work useful in your research, please consider citing:

Wu, H.; Niyogisubizo, J.; Zhao, K.; Meng, J.; Xi, W.; Li, H.; Pan, Y.; Wei, Y. A Weakly Supervised Learning Method for Cell Detection and Tracking Using Incomplete Initial Annotations. Int. J. Mol. Sci. 2023, 24, 16028. https://doi.org/10.3390/ijms242216028

>[Online Published Paper](https://www.mdpi.com/1422-0067/24/22/16028)
