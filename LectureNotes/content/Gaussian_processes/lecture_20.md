# Lecture 20

## Recap of GPs

Here is a schematic of the key points about Gaussian processes (GP).

```{image} /_images/GP_recap_handdrawn.png
:alt: Handdrawn recap of GPs
:class: bg-primary
:width: 600px
:align: center
```

* The "histogram" of GP draws will have the highest density at $\mu$ (which is $\mu(\xvec)$ in general) and $\approx 2/3$ within $\sigma$ of $\mu$. 
The GP is characterized by a kernel $\kappa$, which is a correlation function that gives the covariance for a multivariate Gaussian.
    * Different kernels vary in smoothness, spread, correlation length (a measure of how far apart inputs are to be uncorrelated).
    * The RBF kernel is a prototype:

    $$
      \kappa_{\rm RBF} = \sigma^2 e^{-(x-x')^2/2\ell^2}
    $$

    It is very smooth (cf. some of the Matern kernels), the spread is given by $\sigma$, and the correlation length by $\ell$.

* Using GPs for interpolation (training points are known precisely) or regression (uncertainties at training points).
    * Given (multidimensional) training data with errors or precise, predict *test data* at intermediate $x$ points or extrapolate
    * Impose structure through the kernel.
* Claim: the data "speak more clearly" for GPs than for parametric regression for which there are basis functions (e.g., fitting a polynomial or a sum of Gaussians).

## Selected exercises from notebook

Here we'll try some of the exercises from [](/notebooks/gaussian-processes/Gaussian_processes_exercises.ipynb).


## Application 1: GP emulator from Higdon et al. paper

"A Bayesian Approach for Parameter Estimation and Prediction using a Computationally Intensive Model" by Higdon et al.

* Bayesian model calibration for nuclear DFT  

## Eigenvector continuation emulators

## Application 2:  