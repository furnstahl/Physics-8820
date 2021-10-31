# Lecture 19

## Gaussian process demo notebook

Let's step through the [](notebooks/Gaussian_processes/demo-GaussianProcesses.ipynb) notebook.

* A stochastic *process* is a collection of random variables (RVs) indexed by time or space. I.e., at each time or at each space point there is a random variable.

* A *Gaussian* process (GP) is a stochastic process with definite relationships (correlations!) between the RVs.
In particular, any finite subset (say at $x_1$, $x_2$, $x_3$) has a multivariate normal distribution.
    * cf. the definitions at the beginning of the notebook
    * A GP is the natural generalization of multivariate random variables to infinite (countably or continous) index sets.
    * They look like random functions, but with characteristic degrees of smoothness, correlation lengths, and ranges.

* A multivariate Gaussian distribution in general is

    $$
      p(\xvec|\muvec,\Sigma) = \frac{1}{\sqrt{\det(2\pi\Sigma)}}
        e^{-\frac{1}{2}(\xvec-\muvec)^\intercal\Sigma^{-1}(\xvec - \muvec)}
    $$

    For the bivariate case:

    $$
      \muvec = \pmatrix{\mu_x\\ \mu_y} \quad\mbox{and}\quad
      \Sigma = \pmatrix{\sigma_x^2 & \rho \sigma_x\sigma_y \\
                        \rho\sigma_x\sigma_y & \sigma_y^2}
            \quad\mbox{with}\ 0 < \rho^2 < 1            
    $$

    and $\Sigma$ is positive definite. In the notebook the case $\sigma_x = sigma_y = \sigma$ is seen to be an ellipse.

* Think of the bivariate case with strong correlations ($|\rho|$ close to one) as belong to two points close together $\Lra$ the smoothness of the function tells us that the lines in the plot in the notebook should be closer to flat (small slope). 

* **Kernels:** these are the covariance functions that, given two points in the $N$-dimensional space, say $\xvec_1$ and $\xvec_2$, return the covariance between $\xvec_1$ and $\xvec_2$.

* Consider these vectors to be one-dimensional for simplicity, so we have $x_1$ and $x_2$. The

    $$
      K_{\rm KBF}(x_1,x_2) = \sigma^2 e^{-(x_1-x_2)^2/2\ell^2}
    $$ 

    Compare this to

    $$
      \Sigma = \sigma^2 \pmatrix{1 & \rho \\ \rho & 1} .
    $$

    The diagonals have $x_1 = x_2$ while $\rho = e^{-(x_1-x_2)^2/2\ell^2}$. 

    * So when $x_1$ and $x_2$ are close compared to $\ell$ then the values of the sample at $x_1$ and $x_2$ are highly correlated.
    * When $x_1$ and $x_2$ are far apart, $\rho \rightrarrow 0$ and they become independent. **So $\ell$ plays the role of a correlation length.**

* Look at the examples for different $l$. What does the RBF Cov. Matrix plot show? This is the covariance matrix!

* GP models for regression
    * First example of using GPs $\Lra$ train on some points, predict elsewhere *with$ error bands. 

* No-core shell model $\hbar\omega$ dependence. 

## Games with Gaussian process websites

Here are three websites with Gaussian process visualizations and some things to try with each.

1. The  [*Sample Size Calculations for Computer Experiments*](https://harario.shinyapps.io/Sample_Size_Shiny/) app provides a sandbox for playing with Gaussian processes. Read the "About" tab first; it includes definitions of parameters used in the GP "correlation family" (which define the covariance). The "Sample Path Plots" tab has an app that lets you draw samples of functions that depend on user selected parameters (actually "hyperparameters") that specify the details of the correlation family.

    * Start with *Sample Path Plots*.
    * Try changing *Correlation length* (switch "No. of realizations" to get new draws).
    * Change "Select correlation family".
        * Note the extra parameter with Matern.
    * See the "About" tab for formulas.

2. The [*Gaussian process regression: a function space perspective*](http://rpradeep.me/gpr/) app by Pradeep Ranganathan "demonstrates how a GP prior is a distribution over functions, and how observing data conditions the prior to obtain the GP posterior." 

    * You should see successive draws from a GP.
    * Try changing the covariance and length scale; does it change as predicted?
    * Now try adding points. What happens?
    * Try adjusting the noise. What happens?

3. The [*Gaussian process regression demo*](http://www.tmpl.fi/gp/ ) app "demonstrates Gaussian process regression with one covariate and a set of different covariance kernels." 

    * Check "Show mean and credible intervals" and "sample independently".
    * Add observations.
    * Add a new process.  


## Brief introduction to GPs from Melendez et al.

Here we use the paper by Melendez et al., [Phys. Rev. C **100**, 044001 (2019)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.100.044001), [arXiv:1904.10581](https://arxiv.org/abs/1904.10581) as an introduction to the practical use of GPs.

* Gaussian processes (GPs) are ofen used for *nonparametric regression*. Cf. fitting a polynomial, where the basis functions are $1, x, x^2,\ldots, x^d$ and the coefficients are the parameters. This is *parametric regression*. So with GPs we do not have the corresponding set of parameters for basis functions. But there are parameters that specify the GP itself.

* Besides regression, the other most common application of GPs are to *interpolation*. To carry out either of these we need to *calibrate*, which means fitting the GP parameters.

* A GP is specified by a mean function $m(x)$ [often written $\mu(x)$] and a positive semidefinite covariance function, also called a kernel, $\kappa(x,x')$ where $x \in \mathbb{R}^d$ and $d$ is the dimension of the parameter space. (We will generally suppress the $d$ dependence of $\kappa$.)
    * If we know $m(x)$ and $\kappa(x,x')$, then a GP $f(x)$ is denoted

        $$
              f(x) \sim \mathcal{GP}[m(x), \kappa(x,x')]
        $$
    
        in the same way a normal distribution is $g(x) \sim \mathcal{N(    \mu,\sigma^2)}$.

* While the definition has continuous, infinite-dimensional $x$, in practice we use a finite number of points:

    $$
       \xvec = \{x_i\}_{i=1}^N \quad\mbox{and}\quad
       \fvec = \{f(x_i)\}_{i=1}^N
    $$

    where $N$ is the number of input "points" (which are actually vectors in general). Defining

    $$
      \mvec = m(\xvec) \in \mathbb{R}^N
      \quad\mbox{and}\quad
      K = \kappa(\xvec,\xvec') \in \mathbb{R}^{N\times N}
    $$

    we can write that any subset of inputs form a multivariate Gaussian:

    $$
     \Lra \fvec | \xvec \sim \mathcal{N}(\mvec,K)
    $$

    which can serve as the definition of a GP. We say that "\fvec is conditional on \xvec".

    * The mean function is the *a priori* "best guess" for $f$. If there are no features, this is often taken to be zero.

    * Our specification of the kernel tells us what $K$ is.

* So how do we use this GP? Let's assume we already known $\thetavec$, the set of hyperparameters. And suppose we know the value of the function $f$ at a set of $\xvec_1$ points \Lra this is our *training set*.

    * Therefore partition the inputs into $N_1$ training an $N_2$ test points (the latter are our predictions):

    $$
      \xvec = [\xvec_1 \xvec_2]^\intercal
      \quad\mbox{and}\quad
      \fvec = f(\xvec) = [\fvec_1 \fvec_2]^\intercal .
    $$

    * There are corresponding vectors $\mvec_1$ and $\mvec_2$ and covariance matrices.

