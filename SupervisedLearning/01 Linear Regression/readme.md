# Linear Regression

### Problem Statement

We have a dataset consisting of *labeled examples* $\{(x_i, y_i)\}_{i=1}^N$, where:
- $N$ is the total number of examples.
- $x_i$ is the $D$-dimensional feature vector for example $i$ (i.e., $i = 1, \ldots, N$).
- $y_i$ is the real-valued target associated with example $i$.

Our goal is to *build a model* $f_{w,b}(x)$ that expresses a *linear combination* of the features from the input $x$: $$f_{w,b}(x) = Xw + b$$

where $w$ is a $D$-dimensional *vector* of parameters and $b$ is a *scalar bias*. The notation $f_{w,b}$ indicates that the model is *parametrized* by the parameters $w$ and $b$.

We will use this model to predict the unknown target $y$ for a given input $x$ as follows:
$$
\hat{y} \leftarrow f_{w,b}(x)
$$
Since different pairs $(w, b)$ can yield different predictions for the same input, our objective is to find the optimal parameters $(w^*, b^*)$.

### Solution

In linear regression, the goal is to determine the optimal weights $w$ and bias $b$ that minimize the difference between predicted values and actual values. We will use the Mean Squared Error (MSE) as our cost function:
$$
MSE = J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - (wx_i + b) \right)^2
$$

The choice of MSE is advantageous because it:
- Has a global minimum.
- Is differentiable everywhere.
- Exaggerates the penalty for larger errors, which helps in optimization.

#### Optimizing the Loss Function

To find the optimal parameters, we can use various optimization methods, including:

- **Ordinary Least Squares (OLS)**: Suitable for small datasets with linearly independent features, but it does not involve iterative learning.
- **Gradient Descent**: An iterative algorithm that adjusts $w$ and $b$ based on the gradient of the cost function to minimize the error.
- **Momentum**: An enhancement to gradient descent that accelerates convergence by considering past gradients.
- **Adam**: An adaptive learning rate optimization algorithm that combines the benefits of two other extensions of stochastic gradient descent.
- **RMSprop**: A method that adjusts the learning rate based on the average of recent gradients.

For our explanation, we will focus on Gradient Descent, which is one of the most widely used optimization algorithms in machine learning.

## Gradient Descent

**Steps**:
1. Initialize $w$ and $b$ randomly.
2. Compute the gradients of the cost function $J(w, b)$ with respect to $w$ and $b$.
3. Update the parameters as follows:
   $$
   w = w - \eta \cdot \frac{\partial J}{\partial w}
   $$
   $$
   b = b - \eta \cdot \frac{\partial J}{\partial b}
   $$
   where $\eta$ is the learning rate.
4. Repeat steps 2-3 until convergence (i.e., until the cost function is minimized) or until a maximum number of iterations (epochs) is reached.

### Mathematical Operations

Let’s define the shapes of our data:

$$
\begin{array}{cccc}
X = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,D} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N,1} & x_{N,2} & \cdots & x_{N,D}
\end{bmatrix} & (N \times D) & \quad & \mathbf{y} = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{bmatrix} & (N \times 1)
\\ \\
\mathbf{w} = \begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_D
\end{bmatrix} & (D \times 1) & \quad & \mathbf{\hat{y}} = \begin{bmatrix}
\hat{y}_1 \\
\hat{y}_2 \\
\vdots \\
\hat{y}_N
\end{bmatrix} & (N \times 1)
\end{array}
$$

The prediction of $y$ is given by:
$$
\hat{y} = f_{w,b}(x) = Xw + b
$$

The cost function is:
$$
MSE = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - (wx_i + b) \right)^2
$$

### Gradient Calculations

**Gradient with respect to $w_j$**:

For each feature $j = 1, \ldots, D$:
$$
\frac{\partial MSE}{\partial w_j} = \frac{2}{N} \sum_{i=1}^{N} -x_{i,j} (y_i - \hat{y}_i) = \frac{2}{N} \sum_{i=1}^{N} -x_{i,j} (\hat{y}_i - y_i)
$$

This can be expressed in matrix form for efficiency:
$$
\frac{\partial MSE}{\partial \mathbf{w}} = \frac{2}{N} X^T (\hat{\mathbf{y}} - \mathbf{y})
$$

**Gradient with respect to $b$**:
$$
\frac{\partial MSE}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} - (y_i - \hat{y}_i) = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

This formalism clarifies the optimization process and illustrates how matrix operations can significantly simplify gradient calculations.
