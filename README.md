# VGCN-IRisk

**(VAE-Guided Convolutional Network for Infection Risk Scoring)**

[![img]((https://github.com/dapao111/virus_transmission/blob/master/utils/VGCN.pdf))]
[![Framework](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
<!-- 可选：添加构建状态、代码覆盖率等徽章 -->

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
