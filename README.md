# TCAE-SVM
Predicting functional non-coding variants based on convolutional
autoencoder and support vector machine by deep transfer learning
# Introduction
We propose TCAE-SVM to predict context-specific functional  non-coding variants, Which uses a convolutional autoencoder to extract features and support vector machine to predict functional non-coding variants. 
# Requirements
    Python 3.8
    Keras == 2.4.0
    numpy >= 1.15.4
    scipy >= 1.2.1
    scikit-learn >= 0.20.3
# Data
Download the data from https://pan.baidu.com/s/1ZdkaYJWXXdTc8uDiw1JA1Q?pwd=6l6l 
It has Generic dataset and MPRA dataset for pretraining and testing.
# Usage
    Rscript --vanilla snptoseq.R HGMD HGMD 500 hg19
    Obtain flanking DNA sequence using chromosome coordinates
    python Pretrain.py
    The pre-train.py file contains pretain_model. The model was pretrained by a general functional non-coding variants 
    with the same number of negative variants as positive variants.
    python Predict.py
    The predict.py file contains prediction of context-specific functional  non-coding variants.
# Supported GPUs
    Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, 
    it will proritize using the GPUs if there exist GPUs.
# Contact
Minglie Li: minglie.li@foxmail.com
# Updates
15/8/2023
