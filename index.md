# Algorithmic differentiation for machine learning
#### or one of the reasons why neural networks got so popular.

Training neural networks (NN) is usually done with stochastic gradient decent (SGD) or an advanced version of SGD with momentum etc. (e.g. Adam).
In frameworks there is an abstraction for the gradient calculation.
It is sufficient, when the user defines how the loss is calculated
and the framework does the calculation of the gradients.

In this project we want to take a closer look into how these gradients are obtained.
To follow the project it is not necessary to know neural networks (NN), 
but the motivation will be drawn from NNs most of the time.

Prerequisites: Solid Python knowledge including NumPy
Knowledge about neural networks is not a requirement.

## (Stochastic) gradient decent

Gradient decent is an algorithm that minimizes a cost function given some parameters.
Take for example the cost function of linear least squares:
$$
    J^{\mathrm{Linear Least Squares}} = \sum_n (y_n - \hat y_n)^2 = \sum_n (y_n - {\mathbf{x_n}}^T\boldsymbol{\theta})^2
$$
where $y_n$ are the true values, $\hat y_n$ the estimates, ${\mathbf{x_n}}$ the observations and $\boldsymbol{\theta}$ are the learnable parameters.
Note that bold symbols (e.g. $\mathbf{x}$) indicate vectors while non-bold symbols indicate scalars (e.g. $x$).

Although this function has a closed form solution, we can apply the gradient decent algorithm
$$
    \boldsymbol{\theta}^{\mathrm{new}}
    = \boldsymbol{\theta}^{\mathrm{old}} - \mu \frac{\partial J}{\partial \boldsymbol{\theta}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^{\mathrm{old}}} 
    = \boldsymbol{\theta}^{\mathrm{old}} - \mu
    \begin{bmatrix} 
        \frac{\partial J}{\partial \theta_0} \\
        \frac{\partial J}{\partial \theta_1}
    \end{bmatrix}
    \bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^{\mathrm{old}}}
$$
So we start with a value for $\boldsymbol{\theta}^{\mathrm{old}}$, calculate the gradient of the cost function w.r.t. each parameter and subtract the gradient from the old value with a learning rate $\mu$.
With a sufficient small learning rate and enough iterations the parameters will converge to the optimal parameters.

For the linear least squares problem is the gradient decent algorithm not recommended, because there exists a closed form solution.
The advantage of gradient decent is that one can calculate the gradient even for cost functions where no closed form solution is known.
Therefore gradient decent works for much more cost functions.

## Chain rule

The most important derivation rule in this project is the chain rule:

$$
   J = f(\hat y),\quad \hat y = g(\theta)
$$
$$
   \frac{\partial J}{\partial \theta} = \frac{\partial J}{\partial \hat y}\frac{\partial \hat y}{\partial \theta}
$$

## Algorithmic Differentiation

> In mathematics and computer algebra, automatic differentiation (AD), 
> also called algorithmic differentiation or computational differentiation,[1][2] 
> is a set of techniques to numerically evaluate the derivative of a function specified 
> by a computer program. 
> AD exploits the fact that every computer program, no matter how complicated, 
> executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, 
> division, etc.) and elementary functions (exp, log, sin, cos, etc.). 
> By applying the chain rule repeatedly to these operations, 
> derivatives of arbitrary order can be computed automatically, 
> accurately to working precision, 
> and using at most a small constant factor more arithmetic operations than the original program.
> 
> Automatic differentiation is neither:
> - Symbolic differentiation, nor
> - Numerical differentiation (the method of finite differences).
>
> Symbolic differentiation can lead to inefficient code and faces the 
> difficulty of converting a computer program into a single expression, 
> while numerical differentiation can introduce round-off errors in the discretization process 
> and cancellation. 
> Both classical methods have problems with calculating higher derivatives, 
> where complexity and errors increase. 
> Finally, both classical methods are slow at computing partial 
> derivatives of a function with respect to many inputs, 
> as is needed for gradient-based optimization algorithms. 
> Automatic differentiation solves all of these problems, 
> at the expense of introducing more software dependencies.[citation needed]
 >
 > -- Wikipedia contributors, "Automatic differentiation," Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/w/index.php?title=Automatic_differentiation&oldid=906103043 (accessed August 6, 2019).


## Forward mode

$$
   J = f(\hat y),\quad \hat y = g(h),\quad h = e(\theta)
$$
$$
   \frac{\partial J}{\partial \theta} = 
       \underbrace{
            \frac{\partial J}{\partial \hat y}
            \underbrace{\left(
                \frac{\partial \hat y}{\partial h}
                \frac{\partial h}{\partial \theta}
            \right)}_{\displaystyle \dot{\hat{y}}}
       }_{\displaystyle \dot{J}}
$$
$\dot{J}$ (output gradient) can be calculated from:
 - $J$ (output), 
 - $\hat{y}$ (input) and 
 - $\dot{\hat{y}}$ (input gradient)

So the calculation of $\dot{J}$ does not directly depend on the input $\theta$.

## Reverse mode

$$
   J = f(\hat y),\quad \hat y = g(h),\quad h = e(\theta)
$$
$$
   \frac{\partial J}{\partial \theta} = 
        \underbrace{
            \underbrace{\left(
                \frac{\partial J}{\partial \hat y}
                \frac{\partial \hat y}{\partial h}
            \right)}_{\displaystyle \bar{h}}
            \frac{\partial h}{\partial \theta}
        }_{\displaystyle \bar{\theta}}
$$
$\bar{\theta}$ (input gradient) can be calculated from:
 - $\theta$ (input), 
 - $h$ (output) and 
 - $\bar{h}$ (output gradient)

So the calculation of $\bar{\theta}$ does not directly depend on the loss $J$.

Note the different notation between forward and reverse mode:
 - Forward mode: $\dot{h} = \dfrac{\partial h}{\partial \theta}$
 - Reverse mode: $\bar{\theta} = \dfrac{\partial J}{\partial h}$

## Forward mode: multi input:
$$
   J = f(\hat y),\quad \hat y = g(h, \theta)
$$
$$
   \frac{\partial J}{\partial i} = 
       \underbrace{
            \frac{\partial J}{\partial \hat y}
            \underbrace{\left(
                \frac{\partial \hat y}{\partial h}
                \underbrace{
                    \frac{\partial h}{\partial i}
                }_{\displaystyle\dot{h}_{\mathrm{seed}}}
                +
                \frac{\partial \hat y}{\partial \theta}
                \underbrace{
                    \frac{\partial \theta}{\partial i}
                }_{\displaystyle\dot{\theta}_{\mathrm{seed}}}
            \right)}_{\displaystyle \dot{\hat{y}}}
       }_{\displaystyle \dot{J}}
$$

In case of multiple inputs we use seed values for the inputs to indicate in which derivative we are interested:
 - $\dot{h}_{\mathrm{seed}} = 1, \quad \dot{\theta}_{\mathrm{seed}} = 0$ 
   - -> $\dot{J} = \frac{\partial J}{\partial h}$ 
 - $\dot{h}_{\mathrm{seed}} = 0, \quad \dot{\theta}_{\mathrm{seed}} = 1$
   - -> $\dot{J} = \frac{\partial J}{\partial \theta}$
 - $\dot{h}_{\mathrm{seed}} = 1, \quad \dot{\theta}_{\mathrm{seed}} =  1$
   - -> $\dot{J} = ?$
## NN example

```python
model = Sequential([
    LinearLayer(in_size, hidden_size),
    relu,
    LinearLayer(hidden_size, out_size),
])

feature = ...
log_posterior = model(feature)
posterior = softmax(log_posterior)
loss = binary_cross_entropy(posterior, label)

gradients = loss.backward()
optimizer(model.variables, gradients)
```

![tikz/mlp.svg](tikz/mlp.svg)






