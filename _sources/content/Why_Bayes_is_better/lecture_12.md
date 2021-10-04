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

* In method 1, the function $p(x | D,I)$ weights $f(x)$ at $x_i$ by multiplying by the value of $p(x_i|D,I$), so strong weighting near the peak of $p$ and weak weighting far away. The amount of the weighting is given by the height of the bar. 
In method 2, we have similar weighting of $f(x)$ near to and far from the peak of $p$, but instead of this being accomplished by multiplying by $p(x_j|D,I)$, there are more $x_j$ values near the peak than far away, in proportion to $p(x_j|D,I)$. In the end it is the same weight!

### Marginalization

* Return to the expectation value of $x$ in Eq. {eq}`f_expectation`, but now suppose we introduce a nuisance parameter $y$ into the distribution:

    $$
      p(x|D,I) = \int_{0}^{\infty} dy\, p(x,y|D,I)
    $$ (y_marg)

* We sample $p(x,y|D,I)$ using MCMC to obtain samples $\{(x_j,y_j)\}$, $j=0,\ldots,N'-1$.

* If we had a function of $x$ and $y$, say $g(x,y)$, that we wanted to take the expectation value, we could use our samples:

    $$
      \langle g \rangle \approx \frac{1}{N'}\sum_{j=0}^{N'-1} g(x_j,y_j) .
    $$

* But suppose we just have $f(x)$, so we want to integrate out the $y$ dependence; e.g., going backwards in {eq}`y_marg`? How do we do that with our samples $\{(x_j,y_j)\}$?

    $$\begin{align}
      \langle f \rangle &= \int_{0}^\infty dx f(x)\int_{0}^{\infty} dy\, p(x,y|D,I) \\
      &\approx \frac{1}{N'}\sum_{j=0}^{N'-1} f(x_j) .
    \end{align}$$

* So we just use the $x$ values in each $(x_j,y_j)$ pair. I.e., we  ignore the $y_j$ column!




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
    * Note BDA-3 Figure 11.1. a) is not converged; b) has 1000 iterations and is possibly converged; c) (correlated) draws from the target distribution.
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

    * To get the chains  
    