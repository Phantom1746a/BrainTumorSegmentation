# Brain Tumor Segmentation using U-Net on BraTS Dataset

This repository contains an implementation of brain tumor segmentation using the U-Net architecture on the BraTS (Brain Tumor Segmentation) dataset. The code is organized into modular Python scripts for data loading, model definition, training, testing, and utilities.

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)


---

## Overview

The goal of this project is to segment brain tumors from MRI scans using the U-Net architecture. The BraTS dataset provides multi-modal MRI scans (T1, T1ce, T2, FLAIR) and corresponding ground truth labels for tumor regions. The U-Net model is trained to predict three tumor sub-regions:
- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

---

## Repository Structure
brats_unet/
- ├── data_loading.py # Script for loading and preprocessing BraTS data
- ├── model.py # U-Net model implementation
- ├── train.py # Script for training the U-Net model
- ├── test.py # Script for testing the trained model
- ├── utils.py # Utility functions (e.g., loss calculation, visualization)
- ├── config.py # Configuration file (hyperparameters, paths, etc.)
- ├── dataset.yaml # YAML configuration file
