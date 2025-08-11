# VGCN-IRisk

**(VAE-Guided Convolutional Network for Infection Risk Scoring)**

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
```tree
├── data/
│ ├── data_2025_03_19_new_hamming/ 
│ ├── data_2025_07_22/ 
│ └── data_VAE/  
├── error_bar.py  
├── main_VAE.py 
├── model/ 
│ ├── dataset.py 
│ ├── function.py
│ ├── network.py
│ ├── network_VAE.py
│ ├── network_VAE_decoderconc.py  
│ ├── network_class.py 
│ ├── network_class_ori.py 
│ ├── trainer.py  
│ ├── trainer_5fold.py  
│ └── trainer_VAE.py  
├── predict_sas.py 
├── requirements/ 
├── res_VAE/ 
├── sas_predict_results/ 
│ ├── likely.txt  
│ ├── observed.txt  
│ ├── receptor.txt 
│ └── unlikely.txt  
└── utils/
├── VGCN.pdf 
├── VGCN.svg 
└── config.py  
```
## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dapao111/virus_transmission.git
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
