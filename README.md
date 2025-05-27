# Plant Disease Prediction Using CNN-IDML

## Introduction

This project develops an efficient and robust system for predicting plant diseases using deep learning techniques. By leveraging Convolutional Neural Networks (CNNs) and Introspective Deep Metric Learning (IDML), it addresses critical challenges in agriculture, such as early disease detection, which can significantly impact crop yields and food security. The system is designed to assist farmers, particularly in regions with limited access to experts, by enabling rapid diagnosis via mobile devices.

## Objectives

- Develop a baseline CNN model for plant disease classification using ResNet-50.
- Enhance classification accuracy with an advanced IDML model to handle image uncertainty and variability.
- Mitigate class imbalance using data augmentation and weighted sampling techniques.

## Dataset

The project utilizes the Cassava Leaf Disease Classification dataset from Kaggle, comprising 21,367 labeled images across five classes:

- **Healthy**: Natural green leaves with no disease symptoms.
- **Cassava Bacterial Blight (CBB)**: Leaves with water-soaked lesions and necrotic spots.
- **Cassava Brown Streak Disease (CBSD)**: Brown streaks and yellowing leaves.
- **Cassava Mosaic Disease (CMD)**: Mosaic patterns and chlorosis.
- **Cassava Green Mottle (CGM)**: Mottled green patterns and mild chlorosis.

### Preprocessing

- **Resizing**: Images resized to 224x224 pixels.
- **Augmentation**: Rotation, flipping, color jitter, brightness adjustments, and zoom to enhance dataset variability.
- **Normalization**: Pixel values scaled to [0, 1].
- **Class Imbalance**: Addressed using WeightedRandomSampler and augmentation to balance underrepresented classes.

## Methodologies

Two models were implemented:

1. **Baseline CNN Model**:
   - Backbone: ResNet-50 pretrained on ImageNet.
   - Loss Function: Cross-Entropy Loss.
   - Purpose: Multi-class classification of cassava leaf diseases.
2. **IDML Model**:
   - Backbone: ResNet-50.
   - Loss Function: Proxy-Anchor Loss with introspective similarity metric.
   - Features: Generates semantic and uncertainty embeddings to handle image ambiguity and improve robustness.

## Results

- **Baseline CNN Model**: Validation accuracy of **79.30%**.
- **IDML Model**: Validation accuracy of **84.46%**, with additional metrics:
  - **R@1**: 84.46% (top-1 recall).
  - **NMI**: 0.4743 (clustering quality).
  - **RP**: 94.49% (retrieval precision for top-2 predictions).
  - **MC@R**: 90.11% (mean class precision at top-2 retrieval).

The IDML model outperforms the baseline by effectively managing real-world image variability, making it a promising tool for practical deployment.

## Usage

To reproduce or use this project, follow these steps:

### Prerequisites

- Python 3.8+
- PyTorch
- Required libraries: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`




## Download the Dataset:

Obtain the Cassava Leaf Disease Classification dataset from Kaggle.

## Future Work

Investigate advanced architectures to improve clustering performance (e.g., higher NMI).
Expand the dataset with more diverse images for better generalization.
Optimize the model for real-time inference on low-resource devices.

## Acknowledgements

Dataset: Sourced from the Cassava Leaf Disease Classification Kaggle competition.
Supervisor: Dr. Rajen Kumar Sinha for invaluable guidance.
Authors: Suvramoy Pal and Chandresh Joshi, MSc students at IIT Guwahati.

## License
This project is released under the MIT License.

## Contact
For questions or collaboration, reach out to:

Suvramoy Pal: suvramoy58@gmail.com




