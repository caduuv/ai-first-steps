# Module 4 — Exercises

## Conceptual Questions

### Exercise 4.1: Convolution Dimensions
An image of size 64×64 with 3 channels is passed through:
1. Conv2d(3, 32, kernel_size=5, stride=1, padding=2) — What's the output size?
2. MaxPool2d(2) — What's the output size now?
3. Conv2d(32, 64, kernel_size=3, stride=2, padding=1) — Output size?
4. How many learnable parameters in step 1? (Don't forget biases!)

### Exercise 4.2: Why CNNs?
1. Why do CNNs outperform fully-connected networks on images?
2. What is "parameter sharing" in convolution and why does it matter?
3. What does translation invariance mean and how do CNNs achieve it?

### Exercise 4.3: Augmentation Strategy
For each scenario, design an augmentation pipeline and explain your choices:
1. Classifying handwritten digits (like ZIP codes)
2. Detecting tumors in MRI brain scans
3. Identifying species of flowers from photographs

---

## Implementation Exercises

### Exercise 4.4: Custom Kernels
1. Implement a Gaussian blur kernel of size 5×5
2. Implement an emboss kernel
3. Apply them to an MNIST image and visualize the results
4. Chain two convolutions and compare with their combined kernel

### Exercise 4.5: CNN for Fashion-MNIST
1. Replace MNIST with Fashion-MNIST (same dimensions, harder task)
2. Modify the CNN architecture to improve accuracy
3. Experiment with: number of filters, kernel sizes, dropout rates
4. Report the best configuration and test accuracy

### Exercise 4.6: Transfer Learning
1. Load a pretrained ResNet18 from torchvision
2. Replace the final layer for 10-class CIFAR-10
3. Fine-tune only the last layer (freeze everything else)
4. Compare accuracy with training from scratch
5. Then unfreeze all layers and fine-tune the entire network — compare again

---

## Challenge Exercise

### Exercise 4.7: Grad-CAM Visualization
Implement Grad-CAM (Gradient-weighted Class Activation Mapping):
1. Hook into the last convolutional layer to capture gradients
2. Compute the weighted sum of feature maps
3. Overlay the heatmap on the original image
4. Show which parts of the image the model "looks at" for each class

This is crucial for medical imaging — doctors need to understand WHY a model made its prediction!
