# Lecture 10

We continue with more examples of why Bayes is better! 

## Nuisance parameters (II)

Here we return to the standard problem of fitting a straight line, this time for a real physics case: velocities (the $y$ variables) and distances (the $x$ variables) for a set of galaxies.
* A constant standard deviation of $\sigma = 200\,\mbox{km/sec}$ is given for the $y$ values and no error is given for $x$.
* The question: What value and error should we adopt for the Hubble constant (the slope), assuming we believe that a straight line is a valid model? Further, we don't care about the intercept; indeed, the model is $v = H_0 x$. So the intercept will be a *nuisance parameter*.

We'll compare three estimates in this notebook:
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb)
The three approaches are:
1. Maximum likelihood estimate
1. Single-parameter inference
1. Full Bayesian analysis

Work through the notebook. What follows are some summary notes.

### Bayesian workflow: statistical model

We will be developing a Bayesian workflow as we proceed through the course. A key step will be to make a statistical model.
But rather than defer this step fully to part 3. Full Bayesian analysis, we will give some of the details up front to use them in the other approaches but return to the particularly Bayesian framing below.

For the model we follow the procedure outlined in [parameter_estimation_fitting_straight_line_I.ipynb](../Parameter_estimation/parameter_estimation_fitting_straight_line_I.ipynb).
Our theoretical model $y_M(x)$ is a straight line,

$$
y_M(x) = mx + b
$$

with the parameter vector $\thetavec$ to be determined:

$$
\thetavec = [b, m].
$$

The data here has simple error bars, which is associated with a normal (Gaussian) distribution that is independent for each point. So each data point has a distribution about a mean that is the "true" straight line (at this stage there is no theoretical error assumed, so in the absence of the experimental error the data and theory should match). In particular,

$$
y_i \sim \mathcal{N}(y_M(x_i;\thetavec), \sigma)
$$

or, in other words,

$$
p(y_i\mid x_i,\thetavec) = \frac{1}{\sqrt{2\pi\varepsilon_i^2}} \exp\left(\frac{-\left[y_i - y_M(x_i;\thetavec)\right]^2}{2\varepsilon_i^2}\right)
$$

where $\varepsilon_i$ are the (known) measurement errors indicated by the error bars.
Given the assumption of independent error, the likelihood is the product of likelihoods for each point:

$$
p(D\mid\thetavec) = \prod_{i=1}^N p(x_i,y_i\mid\thetavec) .
$$



### 1. Maximum likelihood estimate (MLE)

* Write the log likelihood based on the expressions from the last part:

$$
 \log(p(D|\thetavec)) = -\frac{1}{2}\sum_{i=1}^N
   \left(\log(2\pi\epsilon_i^2) + \frac{\bigl(y_i - y_m(x_i;\thetavec)\bigr)^2}{\epsilon_i^2}\right)
$$

* Use `scipy.optimize`, obtaining a covariance matrix for the errors in the fit (more on this later).
* Here $\thetavec = [b, H_0]$.
* Result: $b = -26.7 \pm 136.6$ and $H_0 = 80.5 \pm 7.4$
* We note that the result for the intercept is consistent with zero within the uncertainty. As already noted, the actual model we have in mind is $v = H_0 x$, so $b=0$. What does it mean that we allow for a non-zero $b$? 
This is part of our statistical model, with $b$ being associated with either a theoretical or experimental uncertainty (a systematic error in particular).

### 2. Single-parameter inference

* The idea here is that since we don't care about $b$, maybe we should fix it to the maximum likelihood estimate (MLE), leaving a one-parameter fitting problem to solve.

* We calculate the likelihood given this value, and maximize it and find the 68% uncertainty estimate:
  $H_0 = 80.5 \pm 3.8$

* We expect that this approach will *underestimate* the slope uncertainty, because we have assumed that the intercept is known precisely rather than allow for some slop. 
The uncertainty is about half what it was in the full MLE estimate.

### 3. Full Bayesian analysis

* We start with our goal, which is to find $H_0$ given the data.
We use the statistical model described earlier, which means that we introduce the intercept $b$ as a nuisance parameter.
For now we include it as part of $\thetavec$ and seek $p(\thetavec|D,I)$.

* We apply Bayes' theorem to write the desired pdf as being proportional to the likelihood $p(D\mid\thetavec)$ times the
prior $p(\thetavec|I)$ (and the normalization from the denominator doesn't affect our sampling).

* We have the same statistical model, so it leads us to the same likelihood as above.

* For the $\thetavec$ priors we take the symmetric (scale invariant) prior for the slope (recall the earlier discussion of how a uniform distribution in $m$ will bias the slope toward steep values) and a normal distribution with mean zero and standard deviation $\sigma = 200$ for the intercept.

* Use `emcee` to do the sampling.
    * What happened in the posterior plot? $\Lra$ chains take a while to converge $\Lra$ this is the warm-up or burn-in time we need to skip. In this case we started with the slope $m = H_0$ equal to zero, which is far from the equilibrated region.
    * Look at the traces of individual MCMC chains for $b$ and $m$.
    * Choose a warm-up skip of 200 to be conservative.
    * Plot 1,2,3 sigma levels; the smoothness of the contour level curves reflects how many MCMC samples we use (use more for smoother contour lines).

* Next we need to eliminate the nuisance parameter $b$ by marginalization. Marginalization over a nuisance parameter is extremely simple with MCMC.
    * Want $p(\theta_1|D,I) = \int d\theta_0\, p(\theta_0,\theta_1 | D,I)$
    * The output from the MCMC sampler is an array of $N$ samples (after we have discarded the warm-up samples). In this case, the first column consists of the $N$ values of $\theta_0 = b$ and the second column consists of the $N$ values of $\theta_1 = m = H_0$. 
    * To marginalize over $\theta_0$, simply ignore that column and keep the $\theta_{1,i}$ samples!
    * In the code, these samples are given by 
    `slope_samples = emcee_trace[1,:]` and the mean and standard 68% intervals are directly calculated from these.

* The summary of results for $H_0$ is

$$\begin{align}
  \mbox{MLE (1$\sigma$):}&\ \ 80.5 \pm 7.4 \\
  \mbox{Fixed intercept (1$\sigma$):}&\ \ 80.5 \pm 3.8 \\
  \mbox{Bayesian (68% credible regions):}&\ \ 78.2\ (6.8,-6.3)
\end{align}$$ 

* The Bayesian result plotted in the notebook does not appear consistent with the data: the 68% regions do not intercept the error bars 68% of the time (more like 50%) and some points are way off. As part of our Bayesian model checking step (more later!) we would follow-up. Perhaps the data errors are underestimated, or there are outliers, or the theory error is underestimated.



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

More precisely for the Bayesian formulation (choosing a definite case), given $v_{\text measured} = (100\pm 5) \times 10^3\,$km/sec and the posterior for $H_0$, what is the posterior pdf for the distance to the galaxy?

The implementation is in Step 4 of the notebook:
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb#step-4-error-propagation)

The error is calculated two ways:
1. Using the fixed value of $H_0$ from the mean of the previous analysis.
2. Using the full sampled posterior.

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

* Here we need to introduce information on $H_0$. This could be a function (e.g., a specified normal distribution) or just a set of samples $\{H_0^{(i)}\}_{i=1}^N$ generated by our MCMC sampler.
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

