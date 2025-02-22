# Simple Example Calculation for Linear Regression

Consider the following dataset:

| $x$ (input) | $y$ (output) |
|-------------|--------------|
| 1           | 2            |
| 2           | 4            |
| 3           | 6            |

We have a simple linear regression model:

$$ f_{w,b}(x) = wx + b $$

Let's assume the initial parameters are $w = 1$ and $b = 0$.

### Forward Pass Calculation

For each data point, we compute the predicted output $f_{w,b}(x)$:

1. For $x = 1$:
   $$ f_{w,b}(1) = 1 \times 1 + 0 = 1 $$

2. For $x = 2$:
   $$ f_{w,b}(2) = 1 \times 2 + 0 = 2 $$

3. For $x = 3$:
   $$ f_{w,b}(3) = 1 \times 3 + 0 = 3 $$

### Cost Function (Mean Squared Error)

We compute the cost using the Mean Squared Error (MSE) formula:

$$ J(w,b) = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

For this dataset with $m = 3$:

1. For $x = 1$, $y = 2$: 
   $$ (f_{w,b}(1) - y) = (1 - 2) = -1 $$
   $$ (-1)^2 = 1 $$

2. For $x = 2$, $y = 4$:
   $$ (f_{w,b}(2) - y) = (2 - 4) = -2 $$
   $$ (-2)^2 = 4 $$

3. For $x = 3$, $y = 6$:
   $$ (f_{w,b}(3) - y) = (3 - 6) = -3 $$
   $$ (-3)^2 = 9 $$

Now, calculate the MSE:

$$ J(w,b) = \frac{1}{3} \times (1 + 4 + 9) = \frac{14}{3} \approx 4.67 $$

This is the cost for the current parameters.

### Gradient Computation (Backward Pass)

Next, we compute the gradients of the cost function with respect to $w$ and $b$.

The gradients are:

$$ \frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) $$

For $m = 3$:

$$ \frac{\partial J(w,b)}{\partial b} = \frac{1}{3} \times (-1 + -2 + -3) = \frac{-6}{3} = -2 $$

And for the gradient with respect to $w$:

$$ \frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)} $$

For $m = 3$:

$$ \frac{\partial J(w,b)}{\partial w} = \frac{1}{3} \times (-1 \times 1 + -2 \times 2 + -3 \times 3) = \frac{-14}{3} \approx -4.67 $$

### Parameter Update (Training Process)

Using a learning rate $\alpha = 0.1$, we update $w$ and $b$:

- Update $b$:
  $$ b \leftarrow b - \alpha \frac{\partial J}{\partial b} = 0 - 0.1 \times (-2) = 0 + 0.2 = 0.2 $$

- Update $w$:
  $$ w \leftarrow w - \alpha \frac{\partial J}{\partial w} = 1 - 0.1 \times (-4.67) = 1 + 0.467 = 1.467 $$

Thus, the updated parameters are $w = 1.467$ and $b = 0.2$.

### Conclusion

This simple example demonstrates how to compute the forward pass, cost function, gradients, and update the parameters during training in linear regression. The model will continue iterating to minimize the cost function and improve its predictions.
