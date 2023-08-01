# HD-MS-Lesion-Detection

This repository includes a CGAN (conditional generative adversarial network) based framework for synthetic data generation that can be used to create virtual patient image data for improving the performance of AI models. 
We trained the CGAN model specifically for the task of generating synthetic FLAIR images with MS lesions. The repository also includes scripts for training and validation of 3D CNN models (a custom CNN model with attention layer and ResNet18) for the task of new lesion detection on FLAIR MRI. 


## Synthetic Data Generation 

![](/figures/synthetic_data_gen_final.png)

The 'synthetic data generation' folder contains jupyter notebooks for training the CGAN model and use it to a generate synthetic dataset.
1. 'gan_training.ipynb' includes the CGAN training routine .
2. 'ms_lesion_manipulation.ipynb' copies images from the in-house dataset to an external dataset(OASIS) and performs manipulations on the MS segmentaion maps copied from the in-house exams.
3. 'oasis_registration.ipynb' registers the T1 images in the OASIS dataset to the T1 image copied from the in-house dataset.
4. 'merge-segmentations.ipynb' merges the brain segmentaion (generated using fsl FAST algorithm) of the registered T1 image and the MS lesion segmentation.
5. 'inference.ipynb' generates synthetic FLAIR exams using the registered T1 and MS lesion segmentation as input.

## Training procedure for CGAN and classifier networks

![](/figures/training_pipeline_final.png)

The python scripts 'training_resnet_model.py' and 'training_attention_model.py' can be used to train 3D CNN networks (classification models) for the task of detecting new MS lesions. The trained models can be used for prediction using the 'inference.py' script. 

## Steps to build docker image

All the dependencies and python packages for the framework can be set up in a docker image using the Dockerfile.

1. Clone the repository and unzip the contents.
```
git clone https://github.com/NeuroAI-HD/HD-MS-Lesion-Detection.git
```

2. Build a docker image using the following command.
```
cd HD-MS-Lesion-Detection
docker build -t <image_name>:<tag> .
```
