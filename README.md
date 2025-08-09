# VGCN-IRisk

**(VAE-Guided Convolutional Network for Infection Risk Scoring)**

[![img]((https://github.com/dapao111/virus_transmission/blob/master/utils/VGCN.pdf))]
[![Framework](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2F%20TensorFlow-red)]
<img width="2018" height="1114" alt="image" src="https://github.com/user-attachments/assets/2731e9be-13c7-4b0c-a22a-35021d5757d5" />



## 🧬 Overview

This repository contains the official implementation of **[VGCN-IRisk]**, a **hybrid deep learning framework** combining a **Variational Autoencoder (VAE)** and a **Convolutional Neural Network (CNN)** for predicting the probability of viral infection in potential hosts.

*   **Core Idea:** The VAE learns a compressed, probabilistic latent representation of viral genetic similarity features. The CNN then leverages this rich latent representation, along with host-related features, to perform the classification task of predicting infection probability.
*   **Key Benefits:**
    *   Handles high-dimensional and potentially noisy biological data effectively.
    *   Captures uncertainty through the probabilistic VAE latent space.
    *   Extracts hierarchical spatial/sequential patterns via CNN.
    *   Provides a continuous probability score for infection risk assessment.
*   **Target Application:** Virus-host interaction studies, emerging virus risk prediction, host range analysis, viromics.

## 📦 Repository Structure
├── data
│   ├── data_2025_03_19_new_hamming
│   │   ├── SARS_hamming_vector.csv
│   │   ├── train_0_hh.csv
│   │   ├── train_0_vh.csv
│   │   ├── train_0_vv.csv
│   │   ├── valid_0_hh.csv
│   │   ├── valid_0_vh.csv
│   │   └── valid_0_vv.csv
│   ├── data_2025_07_22
│   │   ├── train_hh_e.csv
│   │   ├── train_vh_e.csv
│   │   ├── train_vv_e.csv
│   │   ├── valid_hh_e.csv
│   │   ├── valid_vh_e.csv
│   │   └── valid_vv_e.csv
│   └── data_VAE
│       ├── SARS_association_vector.csv
│       ├── SARS_hamming_vector.csv
│       ├── SARS_hosts_vector.csv
│       ├── SARS_viruses_vector.csv
│       ├── association_vector.csv
│       ├── hosts_vector.csv
│       └── viruses_hamming_vector.csv
├── error_bar.py
├── main_VAE.py
├── model
│   ├── dataset.py
│   ├── function.py
│   ├── network.py
│   ├── network_VAE.py
│   ├── network_VAE_decoderconc.py
│   ├── network_class.py
│   ├── network_class_ori.py
│   ├── trainer.py
│   ├── trainer_5fold.py
│   └── trainer_VAE.py
├── predict_sas.py
├── requirements
├── res_VAE
│   ├── VAE_new_hamming_data_mlpboth_cnn_512_1
│   │   ├── fold_result.csv
│   │   ├── fpr-precision_training_set.pdf
│   │   ├── fpr-precision_xticks_eval.pdf
│   │   ├── idx_label
│   │   │   ├── sampler_train_last_idx_labels.csv
│   │   │   ├── sas_eval_idx_labels.csv
│   │   │   └── test_last_idx_labels.csv
│   │   ├── loss.png
│   │   ├── model
│   │   │   ├── last_swa_model.bin
│   │   │   ├── pytorch_model149.pth
│   │   │   ├── pytorch_model49.pth
│   │   │   └── pytorch_model99.pth
│   │   ├── roc.pdf
│   │   ├── roc_training_set.pdf
│   │   ├── sas_fpr-precision_xticks_eval.pdf
│   │   ├── sas_roc.pdf
│   │   ├── train_ymat_result.csv
│   │   ├── training_loss.csv
│   │   ├── ylabel_result.csv
│   │   ├── ymat_eval_result.csv
│   │   └── ymat_sas_eval_result.csv
│   ├── VAE_new_hamming_data_mlpboth_cnn_512_rs_5
│   ├── VAE_new_hamming_data_mlpboth_cnn_512_rs_7
│   │   └── idx_label
│   └── VAE_new_hamming_data_mlpboth_cnn_64
├── sas_predict_results
│   ├── likely.txt
│   ├── observed.txt
│   ├── receptor.txt
│   └── unlikely.txt
└── utils
    ├── VGCN.pdf
    ├── VGCN.svg
    └── config.py

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_username/your_repo_name.git](https://github.com/dapao111/virus_transmission.git)
    cd virus_transmission
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # OR
    venv\Scripts\activate     # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Ensure Python 3.8+ is installed.*

## 🚀 Training the Model
    ```bash
    python main_VAE.py
    ```
     *Ensure data details is known.*


## 🚀 Predict the Model
    ```bash
    python predict_sas.py
    ```
     *Ensure to be predicted data details is known.*
