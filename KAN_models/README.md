# Taylor-Series-Expanded-Kolmogorov-Arnold-Network-for-Medical-Imaging-Classification
Author : Kaniz Fatema, Emad A. Mohammed, Sukhjit Singh Sehra. 
 Dept. of Applied Computing, Wilfrid Laurier University, Waterloo, Canada
 
This repository provides three lightweight Kolmogorov–Arnold Network (KAN) models for medical image classification with limited data: SBTAYLOR-KAN (B-splines + Taylor series), SBRBF-KAN (B-splines + Gaussian RBFs), and SBWAVELET-KAN (B-splines + Morlet wavelets) for robust, interpretable feature learning.

# 🧠 Kolmogorov–Arnold Networks (KANs) for Medical Image Classification (Small dataset with KAN's best configuration)

This repository presents three novel and lightweight Kolmogorov–Arnold Network (KAN) models developed for **accurate and interpretable classification of medical images**, especially in **data-scarce and resource-constrained clinical environments**.

> **Models Included:**
> - SBTAYLOR-KAN: B-spline + Taylor Series Expansion  
> - SBRBF-KAN: B-spline + Radial Basis Function  
> - SBWAVELET-KAN: B-spline + Morlet Wavelet Transform

---

## 🚀 Highlights

- ✅ Trained **directly on raw images** (no preprocessing or augmentation required)
- ⚙️ Each model uses **~2,872 trainable parameters**, dramatically smaller than traditional deep learning models)
- 🏥 Tailored for **healthcare and clinical deployment** where data is limited
- 📊 Supports **explainable AI** with Grad-CAM visualizations
- 🔁 Generalizes well across diverse datasets (Brain MRI, Chest X-ray, TB, Skin Cancer etc)

---

## 🧩 Model Architectures

### 🔷 1. SBTAYLOR-KAN
- Combines **B-spline basis functions** with a **5-term Taylor series**.
- Taylor expansion models local smooth behavior.
- B-splines capture global non-linear structure.
- Simple yet expressive feature mapping ideal for smooth pattern representation.

### 🔷 2. SBRBF-KAN
- Integrates **Gaussian Radial Basis Functions (RBFs)** with **B-splines**.
- RBFs enable localized feature detection.
- B-splines ensure smooth approximation.
- Uses **Layer Normalization** to maintain stability during training.

### 🔷 3. SBWAVELET-KAN
- Embeds **Morlet wavelet transforms** along with **B-spline embeddings**.
- Captures both **oscillatory (high-frequency)** and **smooth (low-frequency)** features.
- Employs a **softmax-based fusion mechanism** to learn optimal combinations.
- Uses **Batch Normalization** after KAN layers.


