# CROP

A method to accurately predict frameshift-rate (FSR) following a CRISPR-mediated double-strand break (DSB), first proposed in the paper *CROP: A feature-independent context-aware method for CRISPR-Cas9 frameshift prediction*.



## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
- [Usage](#usage)
  - [Predicting with CROP](#predicting-with-crop)
  - [Comparing Against State of the Art Models](#comparing-against-state-of-the-art-models)
- [Model Training Data](#model-training-data)
- [Contact](#contact)

## Introduction

CROP is the first end-to-end architecture for CRISPR-Cas9 repair-outcome prediction that learns biological repair logic directly from target sequences across multiple datasets simultaneously. CROP's output is the distribution of shifts in sequence length following the repair (Δlength), and the frameshift-rate.


## Getting Started

CROP can be run using Docker. While an NVIDIA GPU is recommended for faster performance, CROP will also run on a standard CPU.

### Prerequisites

* **Docker** installed on your system.
* **(Optional) For GPU Support:** An NVIDIA GPU with compatible drivers and the NVIDIA Container Toolkit installed.

## Usage

### Predicting with CROP

CROP expects a `.csv` file containing at least two columns: `Target sequence` and `PAM position`. A sample file `sample.csv` which contains 1,000 samples from the FORECasT K562 dataset [[Allen et al. 2019]](https://www.nature.com/articles/nbt.4317) is provided for testing.

To predict with CROP run:
<details>
<summary><b>Linux / macOS (Bash)</b></summary>
  
```bash
docker run --rm --gpus all \
  --name CROP \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/tensorflow:24.06-tf2-py3 \
  python3 CROP.py sample.csv folder_name
```
</details>
<details> <summary><b>Windows (PowerShell)</b></summary>
  
```powershell
docker run --rm --gpus all `
  --name CROP `
  --ipc=host `
  --ulimit memlock=-1 `
  --ulimit stack=67108864 `
  -v ${PWD}:/workspace `
  -w /workspace `
  nvcr.io/nvidia/tensorflow:24.06-tf2-py3 `
  python3 CROP.py sample.csv folder_name
```
</details>


CROP creates a folder `results/folder_name` and generates a summary of frameshift rates and detailed Δlength files for each cellular-context it was trained on.


---

### Comparing Against State of the Art Models

To predict using the model that we used for benchmarking against state-of-the-art models in Figure 3, run:

<details>
<summary><b>Linux / macOS (Bash)</b></summary>
  
```bash
docker run --rm --gpus all \
  --name CROP \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/tensorflow:24.06-tf2-py3 \
  python3 CROP.py sample.csv folder_name -vs_sota_model
```
</details>
<details> <summary><b>Windows (PowerShell)</b></summary>
  
```powershell
docker run --rm --gpus all `
  --name CROP `
  --ipc=host `
  --ulimit memlock=-1 `
  --ulimit stack=67108864 `
  -v ${PWD}:/workspace `
  -w /workspace `
  nvcr.io/nvidia/tensorflow:24.06-tf2-py3 `
  python3 CROP.py sample.csv folder_name -vs_sota_model
```
</details>

Note, we expect the base model to perform better than this, since this one was trained on less sequences and does not generate predictions based on the SPROUT dataset. For simplicity, the script uses the model created from the first repeat, as from our experience, CROP achieved almost identical results on most datasets. The models for repeats 2-5 are stored in `saved_models_vs_sota/repeats_2_to_5/`.

## Model Training Data 
CROP was trained in 18 datasets (17 for the comparison vs SOTA version) described in the main paper. All datasets used in the paper including those used for training the model and test splits are archived on Figshare at https://doi.org/10.6084/m9.figshare.30998056.

Additionaly, we provide the python notebooks used to preprocess the data, which contain an explanation as to how to obtain the datasets we used. These notebooks are located in the `preprocessing` folder.

## Contact

For issues or questions regarding CROP, please contact tziony.i@gmail.com.
