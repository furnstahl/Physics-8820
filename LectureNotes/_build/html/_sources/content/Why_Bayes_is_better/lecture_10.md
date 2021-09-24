# Lecture 10

We continue with more examples of why Bayes is better! 

## Nuisance parameters (II)

Here we return to the standard problem of fitting a straight line, this time for a real physics case: velocities (the $y$ variables) and distances (the $x$ variables) for a set of galaxies.
* A constant standard deviation of $\sigma = 200\,\mbox{km/sec}$ is given for the $y$ values and no error is given for $x$.
* The question: What value and error should we adopt for the Hubble constant (the slope), assuming we believe that a straight line is a valid model? Further, we don't care about the intercept; indeed, the model is $v = H_0 x$. 

We'll compare three estimates in this notebook:
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb)
The three approaches are:
1. Maximum likelihood estimate
1. Single-parameter inference
1. Full Bayesian analysis

Work through the notebook. What follows are some summary notes.

### 1. Maximum likelihood estimate

* Write the log likelihood:

$$
 \log(p(D|\thetavec)) = -\frac{1}{2}\sum_{i=1}^N
   \left(\log(2\pi\epsilon_i^2) + \frac{\bigl(y_i - y_m(x_i;\thetavec)\bigr)^2}{\epsilon_i^2}\right)
$$

* Use `scipy.optimize`, with covariance matrix for errors (more on this later).
* $\thetavec = [b, H_0]$
* Result: $b = -26.7 \pm 136.6$ and $H_0 = 80.5 \pm 7.4$

### 2. Single-parameter inference

* Since we don't care about $b$, maybe we should fix it to the maximum likelihood estimate (MLE), leaving a one-parameter problem to solve.

* We calculate the likelihood given this value, then the 68% estimate:
  $H_0 = 80.5 \pm 3.8$

* This *underestimates* the slope uncertainty, because we have assumed the intercept is known precisely.

### 3. Full Bayesian analysis

* For priors we take the symmetric (scale invariant) prior for the slope (recall earlier discussion) and a normal distribution with $\sigma = 200$ for the intercept.

* Same likelihood as before.

* Use `emcee` to do the sampling.
    * What happened in the posterior plot? $\Lra$ chains take a while to converge $\Lra$ this is the warm-up or burn-in time we need to skip.
    * Look at the traces of individual MCMC chains for $b$ and $m$.
    * Choose warm-up skip of 200 to be conservative.
    * Plot 1,2,3 sigma levels

* Marginalization is simple with MCMC
    * Want $p(\theta_1|D,I) = \int d\theta_0\, p(\theta_0,\theta_1 | D,I)$



## Error propagation: marginalization

We will use the Hubble constant example to illustrate how we propagate errors.
The set-up is that we have determined a posterior for $H_0$, which is a probability distribution rather than a definite value (i.e., it has uncertainties, which we generically refer to as errors).
Now we want to apply $H_0$ to determine the distance to a galaxy based on applying the equation for the velocity

$$
   v = H_0 x 
$$

to find $x$ given a measurement of $v$. 
$x$ will have uncertainties because of the uncertainty in $v$ *and* the uncertainty in $H_0$.
How do we propagate the error from $H_0$ to $x$  (i.e., combine with the error from $v$)?

More precisely for the Bayesian formulation (choosing a definite case), given $v_{\text measured} = (100\pm 5) \times 10^3\,$km/sec and the posterior for $H_0$, what is the posterior pdf for the distance to the galaxy.

The implementation is in Step 4 of the notebook:
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb#step-4-error-propagation)

The error is calculated two ways:
1. Using the fixed value of $H_0$ from the mean of the previous analysis;
2. using the full sampled posterior.

The statistical model is:
* $v_{\text {exp}} = v_{\text{th}} + \delta v_{\text {exp}}$
* The theory model is $v_{\text {th}} = H_0 x$
* The experimental error in $v$ is assumed to be $\delta v_{\text {exp}} \sim \mathcal{N(0,\sigma_v^2)}$, i.e., it is normally distributed with standard deviation $\sigma_v = 5\times 10^3\,$km/s.
* We assume the measurement error in $v$ (i.e., $\delta v_{\text {exp}}$) is uncorrelated with the error in $H_0$. Note that this should not be an arbitrary assumption but validated by the physics set-up.

Case 1: fixed $H_0$
* Take a uniform prior for $x$: $p(x|I)$ uniform in $[x_{\text {min}},x_{\text {max}}]$.
* Bayes' theorem tells us, with the data $D$ being $v_{\text {exp}}$ and $I$ including $H_0 = \Hhat_0$ and $\delta v_{\text {exp}}$:

$$\begin{align}
  p(x | D, I) &\propto p(D| x, I) p(x |I) \\
              &\propto \frac{1}{\sqrt{2\pi}\sigma_v}
                 e^{-(v_{\text {exp}} - v_{\text {th}})^2/2\sigma_v^2}
                 p(x|I) \\
              &\propto
              \frac{1}{\sqrt{2\pi}\sigma_v}
                 e^{-(v_{\text {exp}} - \Hhat_0 x)^2/2\sigma_v^2}
              \quad\mbox{for}\ x_{\text {min}} < x < x_{\text {max}}
              .
\end{align}$$

Case 2: using the inferred pdf for $H_0$

* Here we need to introduce information on $H_0$. This could be a function (e.g., a specified normal distribution) or just a set of samples $\{H_0^{(i)}}\}_{i=1}^N$ generated by our MCMC sampler.
* As always, we use marginalization (and the product rule, Bayes' rule, etc.).
* The steps are (fill in justifications)

$$\begin{align}
  p(x|D,I) &=  \int dH_0\, p(x,H_0|D,I) \\
    &\propto \int dH_0\, p(D|x,H_0,I) p(x,H_0| I) \\
    & \propto \int dH_0\, p(D|x,H_0,I) p(x|I) p(H_0|I) \\
    & \propto p(x|I) \int dH_0\, p(H_0|I) p(D|x,H_0,I)
\end{align}$$

We have used: marginalization, Bayes' rule, $H_0$ independent of $x$; anything else?
The likelihood $p(D|x,H_0,I)$ was given in case 1 for a particular $H_0$. 

* If we use $H_0$ samples, then the integral over $H_0$ turns in to a simple sum over the marginalized samples:

$$
  p(x | D,I) \approx \frac{1}{N}\sum_{i=1}^{N}
     p(D | x, H_0^{(i)}, I)
$$

See the notebook for the comparison.

## Error propagation: prior information
[Example: Propagation of systematic errors](/notebooks/Why_Bayes_is_better/error_propagation_to_functions_of_uncertain_parameters.ipynb)

