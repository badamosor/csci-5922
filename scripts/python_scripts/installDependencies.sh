#!/bin/sh
#This script install the dependecies of arabic_diacritic_generator.py

#Install nltk

pip install nltk

#Install pytorch

conda install -c anaconda mkl
conda install pytorch-cpu torchvision-cpu -c pytorch
