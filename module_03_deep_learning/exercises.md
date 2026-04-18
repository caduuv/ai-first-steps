# Module 3 — Exercises

## Conceptual Questions

### Exercise 3.1: Activation Functions
1. Why can't we use a linear activation function in hidden layers?
2. What problem does ReLU solve that sigmoid doesn't? What new problem does ReLU introduce ("dying ReLU")?
3. Sketch the graph of Leaky ReLU: `f(x) = max(0.01x, x)`. How does it address dying ReLU?

### Exercise 3.2: Network Architecture
A network has: Input(784) → Hidden(256, ReLU) → Hidden(128, ReLU) → Output(10, Softmax)
1. How many trainable parameters does this network have? (Don't forget biases!)
2. What is the shape of the output?
3. If we double all hidden layer sizes, by how much do parameters increase?

### Exercise 3.3: Backpropagation
For a single neuron: `output = sigmoid(w1*x1 + w2*x2 + b)`
With MSE loss: `L = (output - y)²`
1. Derive ∂L/∂w1 using the chain rule
2. Derive ∂L/∂b
3. If x1=1, x2=2, w1=0.5, w2=-0.3, b=0.1, y=1: compute the numerical value of each gradient

---

## Implementation Exercises

### Exercise 3.4: Implement Leaky ReLU
1. Implement Leaky ReLU and its derivative
2. Modify the NumPy neural network to use Leaky ReLU
3. Compare training on XOR with ReLU vs Leaky ReLU (run each 5 times with different seeds)

### Exercise 3.5: Mini-Batch Gradient Descent
Modify the NumPy neural network to support mini-batch training:
1. Shuffle data at the start of each epoch
2. Split into batches of size 32
3. Compare convergence: full-batch vs mini-batch vs single-sample (SGD)
4. Plot loss curves for all three on the same graph

### Exercise 3.6: Learning Rate Scheduler
Implement a learning rate scheduler that:
1. Starts with `lr = 0.1`
2. Reduces by 50% every 500 epochs
3. Train the network with and without scheduling
4. Compare final accuracies

### Exercise 3.7: PyTorch Regularization
Using PyTorch, implement and compare:
1. L2 regularization via `weight_decay` in the optimizer
2. Dropout layers with different rates (0.1, 0.3, 0.5)
3. BatchNorm layers
4. Combination of all three
Report train/test accuracy for each configuration.

---

## Challenge Exercise

### Exercise 3.8: Multi-Class Neural Network from Scratch
Extend the NumPy neural network to handle multi-class classification:
1. Replace sigmoid output with softmax
2. Replace BCE loss with categorical cross-entropy
3. Derive and implement the backpropagation for softmax + CCE
4. Train on the Iris dataset (3 classes, 4 features)
5. Report accuracy and plot the confusion matrix

*Hint: For softmax + CCE, the gradient simplifies to: `dz = softmax(z) - y_one_hot`*
