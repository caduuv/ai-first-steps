# Module 1 — Exercises

## Conceptual Questions

### Exercise 1.1: Vector Intuition
Given vectors `a = [1, 0, 0]` and `b = [0, 1, 0]`:
1. What is `dot(a, b)`? What does this tell you about their relationship?
2. What is `||a||₂`? What is `||b||₂`?
3. If a neural network neuron computes `output = dot(weights, input) + bias`, what does a zero dot product mean intuitively?

### Exercise 1.2: Matrix Dimensions
A dataset has 1000 images, each 28×28 pixels (flattened to 784 values).
1. What are the dimensions of the data matrix?
2. If a neural network layer has 128 neurons, what are the dimensions of the weight matrix?
3. What is the output dimension after matrix multiplication?

### Exercise 1.3: Probability
A screening test for cervical abnormalities has:
- Sensitivity (P(Positive | Abnormal)) = 0.90
- Specificity (P(Negative | Normal)) = 0.95
- Prevalence (P(Abnormal)) = 0.05

1. Calculate P(Positive) using the law of total probability.
2. Use Bayes' theorem to find P(Abnormal | Positive).
3. Why is this result surprising, and what does it mean for screening programs?

### Exercise 1.4: Loss Functions
1. Why do we use cross-entropy instead of MSE for classification?
2. What happens to binary cross-entropy as the predicted probability approaches 0 when the true label is 1?
3. Implement Huber loss, which combines MSE and MAE. Define it piecewise:
   - If |error| ≤ δ: loss = 0.5 * error²
   - If |error| > δ: loss = δ * (|error| - 0.5 * δ)

---

## Implementation Exercises

### Exercise 1.5: Cosine Similarity
Implement cosine similarity from scratch:
```
cos_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```
Test with:
- Two identical vectors (should be 1.0)
- Two perpendicular vectors (should be 0.0)
- Two opposite vectors (should be -1.0)

### Exercise 1.6: Gradient Descent Variants
Implement **Stochastic Gradient Descent (SGD)** for linear regression:
- Instead of using all data points to compute the gradient, use a random single point
- Compare convergence with full-batch gradient descent
- Then implement **Mini-Batch GD**: use random batches of 32 samples
- Plot loss curves for all three methods on the same graph

### Exercise 1.7: Polynomial Regression
Extend linear regression to fit a polynomial:  `y = ax² + bx + c`
1. Generate data from `y = 2x² - 3x + 1 + noise`
2. Create features `[x², x, 1]` from the input `x`
3. Use the normal equation to find `[a, b, c]`
4. Plot the data and the fitted curve

### Exercise 1.8: Numerical Gradient Checker
Write a function `check_gradients(func, grad_func, test_points)` that:
1. Computes both numerical and analytical gradients at each test point
2. Computes the relative error between them
3. Raises an error if any relative error exceeds 1e-5
4. Test it with both correct and intentionally wrong gradient functions

---

## Challenge Exercise

### Exercise 1.9: Logistic Regression from Scratch
Combine everything in this module to implement **logistic regression**:
1. Model: `P(y=1|x) = sigmoid(w·x + b)`
2. Loss: Binary cross-entropy
3. Optimization: Gradient descent
4. Generate a 2D binary classification dataset
5. Train the model and plot the decision boundary
6. Report accuracy on a held-out test set

*Hint: The gradient of BCE w.r.t. w is: `(1/n) * X.T @ (sigmoid(X @ w + b) - y)`*
