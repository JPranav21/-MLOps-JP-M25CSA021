# MLOps Assignment 1 – MNIST & FashionMNIST Experiments

**Course:** DL-Ops  
**Assignment:** 1  
**Frameworks:** PyTorch, scikit-learn  
**Datasets:** MNIST, FashionMNIST  
**Models:** ResNet-18, ResNet-50, SVM  
**Train–Validation–Test Split:** 70% – 10% – 20%  

---

## Q1(a): Deep Learning Models on MNIST

### Experimental Setup
- Models: **ResNet-18**, **ResNet-50**
- Pretrained Weights: ❌ No
- Optimizers: SGD, Adam
- Learning Rates: 0.001, 0.0001
- Batch Sizes: 16, 32
- Epochs: 5–10
- USE_AMP: True
- Loss Function: Cross-Entropy Loss

---

### MNIST – Test Classification Accuracy (%)

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |
|-----------:|----------:|--------------:|--------------:|--------------:|
| 16 | SGD | 0.001 | **92.4** | 91.1 |
| 16 | SGD | 0.0001 | 89.6 | 88.2 |
| 16 | Adam | 0.001 | **97.8** | 96.9 |
| 16 | Adam | 0.0001 | 95.3 | 94.2 |
| 32 | SGD | 0.001 | 91.2 | 90.1 |
| 32 | SGD | 0.0001 | 88.4 | 87.3 |
| 32 | Adam | 0.001 | **98.1** | 97.4 |
| 32 | Adam | 0.0001 | 96.0 | 95.1 |

---

### Key Observations (MNIST)

- Adam optimizer consistently outperformed SGD
- ResNet-18 performed on par or better than ResNet-50
- Increasing depth did not significantly improve accuracy
- High accuracy achieved even with a small number of epochs
- MNIST is a simple dataset favoring shallow architectures

---

## Q1(b): SVM Classifier Experiments

### Experimental Setup
- Kernels: `rbf`, `poly`
- Regularization Parameter: C = 1.0, 10.0
- Dataset Subset: 10,000 samples
- Feature Representation: Flattened and normalized images

---

### MNIST – SVM Results

| Kernel | C Value | Test Accuracy (%) | Training Time (ms) |
|------:|--------:|------------------:|-------------------:|
| RBF | 1.0 | **96.1** | 1450 |
| RBF | 10.0 | 96.4 | 1980 |
| Poly | 1.0 | 91.8 | **620** |
| Poly | 10.0 | 92.6 | 810 |

---

### Key Observations (SVM)

- RBF kernel achieved the highest accuracy
- Polynomial kernel trained faster but with lower accuracy
- SVMs are computationally expensive for large datasets
- CNNs scale better for image classification tasks

---

## Conclusion

- ResNet-18 provides the best balance between accuracy and efficiency
- Adam optimizer enables faster convergence
- SVMs perform well but are limited by training time
- Deep learning models are better suited for large-scale image datasets

---

## Notes

- All experiments were conducted using PyTorch
- No pretrained weights were used
- Colab notebook contains already executed experiments
- Results are logged in CSV format for reproducibility
