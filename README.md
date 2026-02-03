# Cotton Leaf Disease Diagnosis System 🌿🦠

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Institution](https://img.shields.io/badge/Institution-ESPOL-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Intelligent diagnosis of cotton diseases for sustainable agriculture using Deep Learning.**

## 📖 Overview

Cotton (*Gossypium*) is a crucial crop for rural economies in Ecuador. However, its production is threatened by foliar diseases such as **Curl Virus**, **Leaf Reddening**, and **Leaf Spot Bacterial Blight**, which reduce photosynthetic capacity and fiber quality.

This project implements and compares two Deep Learning architectures to detect these diseases automatically:
1.  **VGG16 (Baseline):** A classic CNN using Transfer Learning.
2.  **SBTAYLOR-KAN modified (Proposed):** A novel hybrid architecture combining Convolutional Neural Networks (CNN) with **Kolmogorov-Arnold Networks (KAN)** and Taylor Series expansion for learnable activation functions.

The proposed **SBTAYLOR-KAN modified** model achieved **94.80% accuracy**, significantly outperforming the baseline while being computationally lighter (only ~241k parameters).

## 🎯 Objectives

* **Early Detection:** Provide a tool for farmers to identify phytosanitary problems early.
* **Optimization:** Reduce the use of chemical treatments by targeting only affected areas.
* **Innovation:** Validate the viability of KANs (Kolmogorov-Arnold Networks) as an alternative to traditional MLPs in computer vision.

## 🧠 Model Architecture: SBTAYLOR-KAN

Unlike traditional CNNs that use fixed activation functions and dense classification layers, our proposed model integrates:

1.  **CNN Block (Feature Extraction):** 4 sub-blocks of Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool (Filters: $32 \rightarrow 64 \rightarrow 128$).
2.  **Adaptive Pooling:** Reduces feature maps to a compact representation ($128 \times 1 \times 1$).
3.  **Taylor Series Function:** Enhances feature representation using 5 terms (odd functions/sine approximation).
4.  **KAN Classifier:** Replaces dense layers with **KANLinear** layers, which use learnable B-splines on edges.
    * *Structure:* $128 \rightarrow 256 \rightarrow 128 \rightarrow 4$ (Classes).


## 📊 Dataset

The model was trained on a composite dataset merging **SAR-CLD-2024** and **COT-AD**, containing images of cotton leaves classified into 4 categories:

* **Fresh Leaf** (Healthy)
* **Curl Virus**
* **Leaf Reddening**
* **Leaf Spot Bacterial Blight**

*Preprocessing:* Images were resized to $224 \times 224$ pixels, normalized, and augmented to ensure robustness.

## 🚀 Key Results

| Metric | VGG16 (Transfer Learning) | SBTAYLOR-KAN (Proposed) |
| :--- | :---: | :---: |
| **Accuracy** | 88.73% | **94.80%** |
| **Parameters** | ~14.8 Million | **~241,536** (Compact) |
| **Inference Time** | Slower | **~300ms** (Real-time ready) |
| **Convergence** | Slow | Fast |

> **Conclusion:** The SBTAYLOR-KAN model offers superior accuracy and clearer decision boundaries (as seen in the confusion matrix) with a fraction of the parameters.

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Deep Learning:** PyTorch (SBTAYLOR-KAN), TensorFlow/Keras (VGG16)
* **Data Processing:** NumPy, Pandas, torchvision
* **Visualization:** Matplotlib, Seaborn, Grad-CAM (for interpretability)
* **Environment:** Jupyter Notebooks / Google Colab

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/cotton-disease-diagnosis.git](https://github.com/yourusername/cotton-disease-diagnosis.git)
    cd cotton-disease-diagnosis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Inference:**
    ```python
    from models import SBTAYLOR_KAN
    import torch

    # Load model
    model = SBTAYLOR_KAN(num_classes=4)
    model.load_state_dict(torch.load('weights/best_model.pth'))
    
    # Predict
    prediction = model.predict('path/to/leaf_image.jpg')
    print(f"Diagnosis: {prediction}")
    ```

## 👥 Authors

**Group 7 - Artificial Intelligence (PAO II 2025)**
* **Michael Estrada Santana** - [GitHub Profile](https://github.com/BryanEstrada003)
* **Melissa Ayllon Gutiérrez** - [GitHub Profile](https://github.com/MelissaAyllon)
* **Juan Pablo Plúas Muñoz** - [GitHub Profile](https://github.com/jppluas)

**Advisor:** Enrique Pelaez  
**Institution:** Escuela Superior Politécnica del Litoral (ESPOL) 🇪🇨

---
References:

[1] P. Bishshash, M. A. S. Nirob, M. H. Shikder, y A. Sarower, «SAR-CLD-2024: A comprehensive dataset for Cotton Leaf Disease Detection». Mendeley Data, 2024.

[2] A. Ali et al., «COT-AD: Cotton Analysis Dataset», arXiv preprint arXiv:2507.18532, 2025.

[3] Anonymous, «Kolmogorov–Arnold Networks: A Critical Assessment of Claims and Empirical Evidence», arXiv preprint arXiv:2407.11075, 2024, [En línea]. Disponible en: https://arxiv.org/abs/2407.11075

[4] K. Fatema, E. A. Mohammed, y S. S. Sehra, «Taylor-Series Expanded Kolmogorov–Arnold Network for Medical Imaging Classification», arXiv preprint arXiv:2509.13687, 2025, [En línea]. Disponible en: https://arxiv.org/abs/2509.13687
