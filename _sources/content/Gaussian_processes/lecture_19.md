# Lecture 19

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