# CROP

A method to accurately predict frameshift-rate (FSR) following a CRISPR-mediated double-strand break (DSB), first proposed in the paper *CROP: A Feature-Independent, Context-Conditioned Model for CRISPR-Cas9 Frameshift Prediction*.



## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Predicting with CROP](#predicting-with-crop)
  - [Comparing Against State of the Art Models](#comparing-against-state-of-the-art-models)
- [Contact](#contact)

## Introduction

CROP is the first end-to-end architecture for CRISPR-Cas9 repair-outcome prediction that learns biological repair logic directly from target sequences across multiple datasets simultaneously. CROP's output is the shift in sequence length following the repair (Δlength).


## Prerequisites

To run CROP with GPU support, ensure you have:

* **NVIDIA GPU** with compatible drivers.
* **Docker** installed with the NVIDIA Container Toolkit.

## Usage

CROP expects a `.csv` file containing at least two columns: `Target sequence` and `PAM position`. A `sample.csv` which contains 1,000 samples from the FORECasT K562 dataset is provided for testing.

To run CROP run:

```bash
docker run --gpus all \
  --name CROP \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/tensorflow:24.06-tf2-py3 \
  python3 CROP.py sample.csv folder_name
```

CROP creates a folder results/folder_name and generates a summary of frameshift rates and detailed Δlength files for each specific model.


---

### Comparing Against State of the Art Models

To run the model using specific configuration that we used for benchmarking against state-of-the-art models in Figure 3, run:

```bash
docker run --gpus all \
  --name CROP \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/tensorflow:24.06-tf2-py3 \
  python3 CROP.py sample.csv folder_name -vs_sota_model
```
Note, we expect the base model to perform better than this, since this one was trained on less sequences and does not predict for the SPROUT dataset.

## Model Training Data 
CROP was trained in 18 datasets (17 for the comparison vs SOTA version) described in the main paper. All datasets used in the paper including those used for training the model and test splits are archived on Figshare at https://doi.org/10.6084/m9.figshare.30998056.

Additionaly, we provide the python notebooks used to preprocess the data, which contain an explanation as to how to obtain the datasets we used. These notebooks are located in the `preprocessing` folder.

## Contact

For issues or questions regarding CROP, please contact tziony.i@gmail.com.
