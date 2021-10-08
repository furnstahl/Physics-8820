# Lecture 13

## Maximum likelihood for least-squares fitting

* Here we consider the problem of fitting a polynomial to *correlated* data. We do this here first with a frequentist approach and come back later to the Bayesian way.

* Our underlying motivation is to get familiar with the linear algebra manipulations.

* We'll use the notation and some of the discussion from Hogg, Bovy, Lang, [arXiv:1008.4686](https://arxiv.org/abs/1008.4686), "Data analysis recipes: Fitting a model to data".

* This paper has 41 extended reference notes, which are worth looking at. Provocative statements about fitting straight lines include:
    > "Let us break with tradition and observe that in almost all cases in which scientists fit a straight line to their data, they are doing something siultaneously *wrong* and *unnecessary.
    "
    * Wrong because it's very rare that a set of two-dimensional measurements are truly drawn from a narrow, linear relationship
    * Probably not linear in detail
    * Unnecessary because communicated slope and intercept are much less informative than the full distribution of data.

* Let's consider fitting $N$ data points to a straight line $y=mx+b$ (here we'll use unbolded capital letters to denote matrices):

    $$
     Y = \pmatrix{y_1 \\ y_2 \\ \vdots \\ y_N}
     \qquad
     A = \pmatrix{1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_N}
     \qquad
     \Sigma = \pmatrix{\sigma_{y_1}^2 & \rho_{12}\sigma_{y_1}\sigma_{y_2} & \cdots & \rho_{1N}\sigma_{y_1}\sigma_{y_N} \\
     & \sigma_{y_2}^2 & \cdots & \cdots \\ 
     & & \ddots & \cdots \\
     & & & \sigma_{y_N}^2 
     }
    $$

    * Here $Y$ is an $N\times 1$ matrix, $A$ is an $N\times 2$ matrix, and $\Sigma$ is an $N\times N$ *symmetric* covariance matrix (because $\Sigma$ is symmetric, we only have to show the upper triangular part).
    * The off-diagonal covariances in $A$ are parametrized by $\rho_{ij}$ in a generalization of the $2\times 2$ form. At this stage they are independent of each other; in the future we'll consider a smooth, function-based variation as $|i - j|$ increases.

* Goal: find $\thetavec = \pmatrix{b \\ m}$ where $Y = A \thetavec$.
(Note that $\thetavec$ is a $2\times 1$ matrix, and the matrix dimensions of the equation for $Y$ work out: $(N\times 1) = (N\times 2)\cdot (2\times 1)$.)

:::{admonition} Check the $N=2$ case
:class: dropdown

$$
  \pmatrix{y_1 \\ y_2} = 
  \pmatrix{1 & x_1 \\ 1 & x_2} \pmatrix{b \\ m}
  = \pmatrix{b+mx_1 \\ b+m x_2}
$$

so each row is $y_i = b + mx_i$, as desired.

:::

:::{note} **Reader:** convince yourself that $N=3$ and higher is correct (e.g., by writing out the $N=3$ case).
:::

:::{admonition} Why don't we just solve the matrix equation $Y = A \thetavec \quad \Lra \quad \thetavec \overset{?}{=}A^{-1}Y$?
:class: dropdown
Because the equation is overconstrained for $N>2$ (ok for $N=2$). What goes wrong?
:::

### Frequentist answer: maximize the likelihood $\propto e^{-\chi^2/2}$

:::{admonition} What is the dimension of $\chi^2$?
:class: dropdown
It is in the exponent by itself, so it must be dimensionless.
:::

* In the familiar, uncorrelated case:

    $$
      \chi^2 = \sum_{i=1}^N \frac{[y_i - f(x_i)]^2}{\sigma_{y_i}^2}
    $$

* In the generalized case:

    $$
     \chi^2 = [Y - A\thetavec]^\intercal\, \Sigmavec^{-1}\, [Y -     A\thetavec]
    $$

    Before going on, make sure you can see how the uncorrelated case is a special case of this expression and how the generalization plays out.

* Check the matrix dimensions on the right side: $(1\times N)\cdot (N\times N)\cdot (N\times 1) \rightarrow 1 \times 1$, which works because $\chi^2$ is a scalar. (If it is confusing how these combine, first write out the matrix products with sums over indices for all matrices. Adjacent indices must run over the same integer values.)
* We have twice used here that $Y - A\thetavec \sim (N\times 1)\cdot (2\times 1) \rightarrow (N\times 1)$. In the first instance the transpose converts $(N\times 1)$ to $(1\times N)$.

**Claim:** the maximum likelihood estimate (MLE) for $\thetavec$ is

$$
  \thetavechat = [A^\intercal \Sigmavec^{-1} A]^{-1}
     [A^\intercal \Sigmavec^{-1} Y] .
$$

:::{admonition} Why can't I say $[A^\intercal \Sigmavec^{-1} A]^{-1}[A^\intercal \Sigmavec^{-1} Y] = A^{-1}\Sigmavec (A^{\intercal})^{-1}A^\intercal \Sigmavec^{-1} Y = A^{-1}Y$? 
:class: dropdown
Because these are not square, invertible matrices, so those operations don't hold.
:::

* Let's make a plausibility argument for the $\thetavechat$ result.
We need square, invertible matrices before we can do the inversion that fails for $Y = A\thetavec$.
    * Start with $Y = A\thetavec$, which has $A$ as an $(N\times 2)$ matrix. 
    * We need to change what multiplies $\thetavec$ into a square matrix.
    * We do this by multiplying both sides by $A^\intercal \Sigmavec^{-1}$, which is $(2\times N)$:

    $$
       A^\intercal \Sigmavec^{-1} Y  = A^\intercal \Sigmavec^{-1} A \thetavec  
    $$

    * Now we can take the inverse to get the result for $\thetavechat$.

* Before proving the results carefully, let's generalize to a higher-order polynomial. E.g., for the quadratic case.

    :::{admonition} If $y_i = q x_i^2 + m x_i + b$, what do     $\thetavec$ and $A$ look like?
    :class: dropdown
    
    $$
      \thetavec = \pmatrix{b \\ m \\ q}
      \quad\mbox{and}\quad
      \pmatrix{1 & x_1 & x_1^2 \\
              \vdots & \vdots & \vdots \\
                1 & x_N  & x_N^2
               }
    $$

    :::   

    The generalization to higher order is straightforward.

* To derive the result for $\thetavechat$, we will write all matrices and vectors with indices, using the Einstein summation convention.  

* Start with

    $$
      \chi^2 = [Y - A\thetavec]^\intercal\, \Sigmavec^{-1}\, [Y - A\thetavec] =
      (Y_i - A_{ij}\thetavec_j)(\Sigma^{-1})_{ii'}(Y_{i'}- A_{i'j'}\thetavec_{j'}) ,
    $$

    where $i,i'$ run from $1$ to $N$ and $j,j'$ run from one to $p$, where the highest power term is $x^{p-1}$. Be sure you understand the indices on the leftmost term, remembering that the matrix expression has this term transposed.

* We find the MLE from $\partial\chi^2/\partial\thetavec_k = 0$ for $k = 1,\ldots p$. 

:::{admonition} Carry out the derivatives.
:class: dropdown

$$\begin{align}
 \left.\frac{\partial\chi^2}{\partial\thetavec_k}\right|_{\thetavec=\thetavechat}
 &= -A_{ij}\delta_{jk}(\Sigma^{-1})_{i,i'}(Y_{i'} - A_{i'j'}\thetavechat_{j'}) + 
 (Y_{i} - A_{ij}\thetavechat_{j})(\Sigma^{-1})_{i,i'}(-A_{i'j'}\delta_{j'k}) = 0
\end{align}$$

:::

* Isolate the $\thetavec$ terms on one side and show the doubled terms are equal:

    $$
     A_{ik}(\Sigmavec^{-1})_{i,i'}Y_{i'}
     + Y_i (\Sigmavec^{-1})_{i,i'} A_{i'k}
     =
     A_{ik}(\Sigmavec^{-1})_{i,i'}A_{i'j'}\thetavechat_{j'}
     + A_{ij}\thetavechat_{j}(\Sigmavec^{-1})_{i,i'}A_{i'k}
    $$   

    * In the second term on the left, switch $i\leftrightarrow i'$ and use $(\Sigmavec^{-1})_{i',i} = (\Sigmavec^{-1})_{i,i'}$ because it is symmetric. This is then the same as the first term.
    * In the first term on the right, we switch $j\leftrightarrow j'$ and use the symmetry of $\Sigmavec$ again to show the two terms are the same. 

* Writing $A_{ik} = (A^\intercal)_{ki}$, we get
    
    $$\begin{align}
     2(A^\intercal)_{ki} (\Sigmavec^{-1})_{i,i'} Y_i
      = 2 (A^{\intercal})_{ki} (\Sigmavec^{-1})_{i,i'} A_    {i'j}\thetavechat_j
    \end{align}$$
    
    or, removing the indices,
    
    $$
      (A^{\intercal}\Sigmavec^{-1} Y) = (A^{\intercal}\Sigmavec^{-1}A)    \thetavechat
    $$
    
    and then inverting (which we showed earlier was possible because     the expression in parentheses on the right is a square, invertible     matrix), we finally obtain:
    
    $$
      \thetavechat = [A^\intercal \Sigmavec^{-1} A]^{-1}
         [A^\intercal \Sigmavec^{-1} Y] .
    $$

    Q.E.D.
    

---

## Dealing with outliers

* Our exploration of different approaches to handling outliers is worked out in [](/notebooks/Why_Bayes_is_better/dealing_with_outliers.ipynb).
    * The details are in the notebook; here we give an overview.

* Our example is linear regressions with data outliers, meaning we fit a linear function (here just a line in one variable) to data that includes one or more points that are many standard deviations from the trend.

* The model setup is familiar by now ($\yth \equiv \ym$):

    $$\begin{align}
      \yexp = \yth + \delta \yexp  \\
      \yth \longrightarrow \ym(x;\thetavec) = \theta_1 x + \theta_0 \\
     \delta\yexp \sim \mathcal{N}(0,\sigma_0^2) 
    \end{align}$$

    :::{Admonition} What is the likelihood $p(\{x_i,y_i\}|\thetavec,\sigma_0)$?
    :class: dropdown

    $$
      p(\{x_i,y_i\}|\thetavec,\sigma_0)
      =\frac{1}{\sqrt{2\pi\sigma_0^2}}
      e^{-(y_i-y_m(x_i;\thetavec))^2/2\sigma_0^2}
    $$

    :::

* Having defined a likelihood, we can apply frequentist methods. Later, when we do the Bayesian approach and we'll need priors, we will take them as uniform (even if that choice is not so well motivated).

### Frequentist: standard likelihood approach and Huber loss

* The standard approach shows the out-sized influence of outliers with a squared-loss function (usual least squares).
* The Huber loss approach switches to a linear loss function for larger deviations (with the crossover parametrized), which reduces the loss contribution of outliers $\Lra$ much more intuitive.
* Some issues are identified.

### Bayesian approaches

1. Conservative model
2. Good-and-bad data model
3. Cauchy formulation
4. Many nuisance parameters.

Step through these and discuss how each one works (and how well). Sivia in chapter 8 has further commentary.


