# Lecture 12

## Integration and marginalization by sampling: intuition

### Integration

* Comparison of integration methods
    1. Approximation by Riemann sum (or trapezoid rule).
    1. Approximation using MCMC samples.

* Suppose we want to calculate the expectation value of a function $f(x)$ where the distribution of $x$ is $p(x|D,I)$ and we'll restrict the domain of $x$ to the positive real axis. So

    $$
       \langle f\rangle = \int_{0}^\infty f(x) p(x|D,I) dx
    $$ (f_expectation)

    and we'll imagine $p(x|D,I)$ is like the red lines in the figures here.

    ```{image} /_images/schematic_histogram.png 
    :alt: schematic histogram
    :class: bg-primary
    :width: 300px
    ```

    ```{image} /_images/pdf_histogram_sampled.png
    :alt: sampled histogram
    :class: bg-primary
    :width: 300px
    ```

* In method 1, we divide the $x$ axis into bins of width $\delta x$, let $x_i = i\cdot\Delta x$, $i=0,\ldots,N-1$. Then

    $$
      \langle f\rangle \approx \sum_{i=0}^{N-1} f(x_i)\, p(x_i|D,I) \,\Delta x .
    $$

    The histogram bar height tells us the strength of $p(x_i|D,I)$ and we have a similar discretization for $f(x)$.

* In method 2, we *sample* $p(x|D,I)$ to obtain a representative set of $x$ values, which we'll denote by $\{x_j\}$, $j=0,\ldots,N'-1$. If I histogram $\{x_j\}$ it looks like the posterior, as seen in the figures. To approximate the expectation value, we simply use:

    $$
      \langle f\rangle \approx \frac{1}{N'}\sum_{j=0}^{N'-1} f(x_j) .
    $$

* In method 1, the function $p(x | D,I)$ weights $f(x)$ at $x_i$ by multiplying by the value of $p(x_i|D,I$), so strong weighting near the peak of $p$ and weak weighting far away. The amount of the weighting is given (approximately) by the height of the corresponding histogram bar. 
In method 2, we have similar weighting of $f(x)$ near to and far from the peak of $p$, but instead of this being accomplished by multiplying by $p(x_j|D,I)$, there are more $x_j$ values near the peak than far away, in proportion to $p(x_j|D,I)$. In the end it is the same weight!

### Marginalization

* Return to the expectation value of $x$ in Eq. {eq}`f_expectation`, but now suppose we introduce a nuisance parameter $y$ into the distribution:

    $$
      p(x|D,I) = \int_{0}^{\infty} dy\, p(x,y|D,I)
    $$ (y_marg)

* And now we sample $p(x,y|D,I)$ using MCMC to obtain samples $\{(x_j,y_j)\}$, $j=0,\ldots,N'-1$.

* If we had a function of $x$ and $y$, say $g(x,y)$, that we wanted to take the expectation value, we could use our samples as usual:

    $$
      \langle g \rangle \approx \frac{1}{N'}\sum_{j=0}^{N'-1} g(x_j,y_j) .
    $$ (g_exp_samp)

* But suppose we just have $f(x)$, so we want to integrate out the $y$ dependence; e.g., going backwards in {eq}`y_marg`? How do we do that with our samples $\{(x_j,y_j)\}$?

    $$\begin{align}
      \langle f \rangle &= \int_{0}^\infty dx f(x)\int_{0}^{\infty} dy\, p(x,y|D,I) \\
      &\approx \frac{1}{N'}\sum_{j=0}^{N'-1} f(x_j) .
    \end{align}$$ (f_exp_samp)

    Equivalently we can note that $f(x)$ is just a special case of $g(x,y)$ with no $y$ dependence. Then {eq}`g_exp_samp` gives the same formula for $\langle f\rangle$ as {eq}`f_exp_samp`.

* So to marginalize, we just use the $x$ values in each $(x_j,y_j)$ pair. **I.e., we ignore the $y_j$ column!**




## MCMC Sampling Interlude: Assessing Convergence

* We've seen that using MCMC with the Metropolis-Hastings algorithm can lead to a Markov chain: a set of confiurations of the parameters we are sampling. This chain enables inference because they are samples of the posterior of interest.

* But how do we know the chain is *converged*? That is, that it is providing samples of the *stationary* distribution? We need diagnostics of convergence!

* What can go wrong?
    * Could fail to have converged.
    * Could exhibit apparent convergence but the full posterior has not been sampled.
    * Convergence could be only approximate because the samples are correlated (need to run longer).

* Strategies to monitor convergence:
    1. Run multiple chains with distributed starting points.
    1. Compute variation between and within chains $\Lra$ look for mixing and stationarity.
    1. Make sure the acceptance rate for MC steps is not too low or too high.

* In this schematic version of Figure 11.3 in BDA-3, we see on the left two chains that stay in separate regions of $\theta_0$ (no mixing) while on the right there is mixing but neither chain shows a stationary distribution (the $\theta_0$ distribution keeps changing with MC steps).

```{image} /_images/schematic_BDA3_fig11p3.png
:alt: schematic BDA3 Figure 11.3
:class: bg-primary
:width: 500px
:align: center
```
* Step through the [MCMC-diagnostics.ipynb](notebooks/MCMC_sampling_I/MCMC-diagnostics.ipynb) notebook, which goes through a laundry list of diagnostics. We'll return to these later in the context of `pymc3`.

* Some notes:
    * In BDA-3 Figure 11.1, a) is not converged; b) has 1000 iterations and is possibly converged; c) shows (correlated) draws from the target distribution.
    * We're doing straight-line fitting again using the `emcee` sampling, but now with the Metropolis-Hasting algorithm (more below on the default algorithm). 
    * In `emcee`, we use `moves.GaussianMove(cov)`, which implements a Metropolis step using a Gaussian proposal with mean zero and covariance `cov`. 

            # MH-Sampler setup
            stepsize = .005
            cov = stepsize * np.eye(ndim)
            p0 = np.random.rand(nwalkers,ndim)
            
            # initialize the sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, dy],
            moves=emcee.moves.GaussianMove(cov))

    * The covariance `cov` could be a scalar, as it is here, or a vector or a matrix. See the relevant [emcee manual page](https://emcee.readthedocs.io/en/stable/user/moves/) for further details and more general moves.

    * The `stepsize` parameter is at our disposal to explore the consequences on convergence of it being too large or too small.

    * To get the chains from the above code snippet we use `sampler.chain`, which will give a list with the shape (# walkers, # steps, # dimensions). So 10 walkers taking 2000 steps each for a two-dimensional posterior (that is, $\thetavec$ has two components) has the shape (10, 2000, 2). We can combine the results from all the walkers with `sampler.chain.reshape((-1,ndim))`, which flattens the first two axes of the list. (One reshape dimension can always be $-1$, which infers the value from the length of the array. So here the reshaped array will have two axes with the second one having dimension `ndim`.)

    * How do we know a chain has converged to a representation of the posterior? **Standard error of the mean $SE(\overline\thetavec)$.**
        * This asks how the *mean* of $\thetavec$ deviates in the chain     from the true distribution mean. Thus it is the simulation (or     sampling) error of the mean, not the underlying uncertainty (    or spread) of $\thetavec$.
        * Calculate it for $N$ samples as
    
        $$
           SE(\overline\thetavec) = \frac{\text{posterior standard     deviation}}{\sqrt{N}}
        $$
    
        * Visualize this with a moving average $\Lra$ check for     stability.
    * Autocorrelation: do you recognize the formula in the code?
    * Acceptance rate. Usually autotuned in packaged MCMC software.

    * Assess the mixing with the *Gelman-Rubin diagnostic*.
        * We'll come back to this later, so this is just a quick pass.
        * Basic idea: multiple chains from different walkers (after warm-up) are split up and one looks at the variance within a chain and between chain.
        * There is some internal documentation in the notebook; see BDA-3 pages 284-5 for more details.

* Try changing the `step_size` in the notebook to see what happens to each of the diagnostics.    

* Point of emphasis: "The key purpose of MCMC is *not* to explore the posterior but to estimate expectation values."

### Figures to make every time you run MCMC (following Hogg and Foreman-Mackey sect. 9)

* Trace plots
    * The burn-in length can be seen; can identify problems with model or sampler; qualitative judge of convergence.
    * Use convergence diagnostic such as Gelman-Rubin.

* Corner plots
    * If you have a $D$-dimensional parameter space, plot all $D$ diagonal and all ${D\choose 2}$ joint histograms to show low-level covariances and non-linearities.
    * "... they are remarkable for locating expected and unexpected parameter relationships, and often invaluable for suggesting re-parameterizations and transformation that simplify your problem."

* Posterior predictive plots 
    * Take $K$ random samples from your chain, plot the prediction each sample makes for the data and over-plot the observed data.
    * "This plot gives a qualitative sense of how well the model fits the data and it can identify problems with sampling or convergence."

### What to do about sampling from correlated distributions?

* If our posterior has projections that are slanted, indicating correlations, and we are doing Metropolis-Hastings (MH) sampling, how do we decide on a step size?
    * The problem is that we want to step differently in different directions: a long enough step size to explore the long axis will lead to many rejections in the orthogonal direction.
    * So we do not want an isotropic step proposal!

* If we propose steps by

    $$
     p(\xvec) = \frac{1}{\sqrt{(2\pi)^N |\Sigma|}}
       e^{-\xvec^{\intercal}\cdot \Sigma^{-1} \cdot \xvec} ,
    $$

    then we don't need to take $\Sigma \propto \sigma^2 \mathbb{1}_N$! 
    * We have $N(N-1)$ parameters to "tune" to reduce the correlation time.
    * However, this is increasingly difficult as $N$ increases.

* One improvement is to do a linear transformation of $\thetavec \longrightarrow \thetavec' = A\thetavec + B$ such that $\thetavec'$ is uncorrelated with similar $\sigma_i$'s in each direction. Thus effectively to rotate the slanted ellipse.

* Or one could use an "affine invariant" sampler such as `emcee`.
    * An affine transformation is an invertible mapping from $\mathbb{R}^N \rightarrow \mathbb{R}^N$, namely $\yvec = A\xvec + B$, which is a combination of stretching, rotation, and translation.
    * Affine invariant means that the sampler performs equally well on all affine tranformations of a distribution.
    * So `emcee` figures out how to make the appropriate steps.
    * It does this by using the many walkers at time $t$, which have sampled the space, to construct an appropriate affine compatible update step for $t+1$. This is one reason to make sure there are plenty of walkers.
    