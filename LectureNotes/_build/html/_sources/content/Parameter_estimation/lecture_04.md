# Lecture 4: Parameter estimation

## Overview comments

* In general terms, "parameter estimation" in physics means obtaining values for parameters (i.e., constants) that appear in a theoretical model that describes data. (Exceptions exist, of course.)
* Examples:
    * couplings in a Hamiltonian
    * coefficients of a polynomial or exponential model of data
* Conventionally this process is known as "fitting the parameters" and the goal is to find the "best fit" and maybe error bars.
* We will make particular interpretations of these phrases from our Bayesian point of view.
* Plan: set up the problem and look at how familiar ideas like "least-squares fitting" show up from a Bayesian perspective.
* As we proceed, we'll make the case that for physics a Bayesian approach is particular well suited.

## What can go wrong in a fit?

As a teaser, let's ask: what can go wrong in a fit? 

```{image} /_images/over_under_fitting_cartoon.png
:alt: bootstrapping
:class: bg-primary
:width: 400px
:align: center
```

Bayesian methods can identify and prevent both underfitting (model is not complex enough to describe the fit data) or overfitting (model tunes to data fluctuations or terms are underdetermined, leading to them playing off each other).  
$\Longrightarrow$ we'll see how this plays out.

## Step through a notebook

Let's step through parameter_estimation_Gaussian_noise.ipynb.

* Import of modules
    * Using seabon just to make nice graphs
    * We'll use emcee here (cf. "MC" $\rightarrow$ "Monte Carlo") to do the sampling.
    * corner is used to make a particular type of plot.
* Example from Sivia's book {cite}`Sivia2006`: Gaussian noise and averages.

    $$
      p(x | \mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}
         e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

    where $\mu$ and $\sigma$ are given and the pdf is normalized ($\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2} dx = 1$).

    ::::{admonition} Question
    What are the dimensions of this pdf?
    :::{admonition} Answer
    :class: dropdown 
    one over length ($1/x$).
    :::
    ::::

    Its justification as a theoretical model is via maximum entropy, the "central limit theorem" (CLT), or general considerations, all of which we will come back to in the future.

* $M$ measurements $D \equiv \{x_k\} = (x_1, \ldots, x_M)$ (e.g., $M=100$), distributed according to $p(x|\mu,\sigma)$ (that implies that if you histogrammed the samples, they would roughly look like the Gaussian).

* How do we get such measurements? As in the Exploring_pdfs.ipynb notebook, we "sample" from $\mathcal{N}(\mu,\sigma^2)$.

* Goal: given the measurements $D$, find the approximate $\mu$ and $\sigma$.
    * Frequentist: use the maximum likelihood method
    * Bayesian: compute posterior pdf $p(\mu,\sigma|D,I)$

* Random seed of 1 means the same series of "random" numbers are used every time you repeat. If you put 2 or 42, then a different series from 1 will be used, but still the same with every run that has that seed.

* `stats.norm.rvs` ("norm" for normal or Gaussian distribution; "rvs" for random variates) as in Exploring_pdfs.ipynb. 
    * `size=M` is a "keyword argument" (often `kw` $\equiv$ keyword), which means it is optional and there is a default value (here the default is $M=1$).

* `shift-tab-tab` after evaluating cell will give you information
    * e.g., put your cursor on "norm" or "rvs" and `shift-tab-tab` will tell you all about these.

* The output $D$ is a numpy array. (Everything in Python is an *object*, so more than just a datatype, there are methods that go with these arrays.)
    * Put your cursor just after `D` and `shift-tab-tab`
    * $[\cdots]$ when printed.

* Consider the number of entries in the tails, say beyond $2\sigma$ $\Longrightarrow$ $x>12$ or $x < 8$.
    * How many do you expect *on average*? $2\sigma$ means about 95\%, so about 5/100.
    * Here there are 4 in that range. If there were zero is there a bug? No, there is a chance that will happen!

* Note the pattern (or lack of patter) and repeat to get different numbers. (How? Change the random seed from 1.) Always play! 

* Questions about plotting? Some notes:
    * We'll repeatedly use constructions like this, so get used to it!
    * `;` means we put multiple statements on the same line; this is not necessary and probably should be avoided in most cases.
    * `alpha=0.5` makes the (default) color lighter.
    * Try `color='red'` on your own in the scatter plot.
    * You might prefer side-by-side graphs $\Longrightarrow$ alternative code.
    * An "axis" in Matplotlib means an entire subfigure, not just the x-axi or y-axis.
    * If you want to know about a potting command already there, `shift-tab-tab` (or you can always google it).
    * To find `vlines` (vertical lines), google "matplotlib vertical line". (Try it to find horizontal lines.)
    * `fig.tight_layout()` for good spacing with subplots.

* Observations on graphs?
    * Scatter plot shows tail $\Longrightarrow$ in this case there *are* 5, but rerun and it will be more or less $\Longrightarrow$ *everything is a pdf*.
    * The histogram is imperfect. Is this a problem? cf. the end of Exploring_pdfs.ipynb with different numbers of samples.
    * Tails fluctuate!

* Frequentist approach
    * *true* value for parameters $\mu,\sigma$, not a pdf
    * Use of $\mathcal{L}$ is common notation for likelihood.
    * Why the product? *Assumed* independent. Reasonable?
    * $\log\mathcal{L}$ for several reasons.
        * to avoid problems with extreme values
        * note: "$\log$" always means $\ln$. If we want base 10 we'll use $\log_10$.
        * $\mathcal{L} = (\text{const.})e^{-\chi^2}$ so maximizing $\mathcal{L}$ is same as maximizing $\log\mathcal{L}$ or minimizing $\chi^2$.

    :::{admonition} Carry out the maximization
    :class: dropdown
    
    $$
      \frac{\partial\log\mathcal{L}}{\partial\mu}
      = -\frac{1}{2}\sum_{i=1}^M 2 \frac{x_i-\mu}{\sigma^2}\cdot (-1)
      = \frac{1}{\sigma^2}\sum_{i=1}^M (x_i-\mu)
      = \frac{1}{\sigma^2}\Bigl(\bigl(\sum_{i=1}^M x_i\bigr) - M\mu\Bigr)
    $$
    
    Set equal to zero to find $\mu_0$ $\Longrightarrow$ 
    $M\mu_0 = \sum_{i=1}^M x_i$ or $\mu_0 = \frac{1}{M}\sum_{i=1}^M x_i$.
    
    You do $\sigma_0^2$! (Easier to do $d/d\sigma^2$ than $d/d\sigma$.) 
    :::

    * Do these make sense?
        * $\mu_0$ is the mean of data $\rightarrow$ *estimator* for "true mean.
        * $\sigma_0"$ gives spread about $\mu_0$.
    * Note the use of `.sum` to add up the $D$ array elements.
    * Printing with f strings: `f'...'`
        * `.2f` means a float with 2 decimal places.
    * Note comment on "unbiased estimator"
        * an *accurate* statistic
        * Here compare $\mu_0$ estimated from $\frac{1}{M}$ vs. $\frac{1}{M-1}$.
        * If you do this many times, you'll find that $\frac{1}{M}$ doesn't quite give $\mu_{\rm true}$ correctly (take mean of $\mu_0$s from many trials) but $\frac{1}{M-1}$ does!
        * The difference is $\mathcal{O}(1/M)$, so small for large $M$.
    * Compare estimates to true. Are they good estimates? How can you tell? E.g., should they be within 0.1, 0.01, or what?
    (More about this as we proceed!)

* Bayesian approach $\Longrightarrow$ $p(\mu,\sigma|D,I)$ is the posterior: the probability (density) of finding some $\mu,\sigma$ given data $D$ and what else we know ($I$).
    * $I$ could be that $\sigma > 0$ or $\mu$ should be near zero.        
:::{admonition} Frequentist probability
Long-run frequency of (real or imagined) trials $\Longrightarrow$ data is probabilistic (repeat experiment and get different result) but model parameters are not (universe stays the same with more observations).
:::

:::{admonition} Bayesian probability
Quantification of information (what you know, often said as "what you believe"). Data are fixed (it's what you found) but knowledge of true model parameters is fuzzy (and gets updated with more trials, cf. coin flipping).
:::

One more time with Bayes' theorem:

$$
  p(\mu,\sigma | D,I) = \frac{p(D | \mu,\sigma,I)\,p(\mu,\sigma|I)}{p(D|I)}
$$ (eq:bayes_again)

:::{admonition} Label each term in Eq. {eq}`eq:bayes_again`.
:class: dropdown

$$
  \underbrace{p(\mu,\sigma | D,I)}_{\text{posterior}} = \frac{\overbrace{p(D | \mu,\sigma,I)}^{\text{likelihood}}\ \ \overbrace{p(\mu,\sigma|I)}^{\text{prior}}}{\underbrace{p(D|I)}_{\text{evidence or data probability}}}
$$

:::

* Tells you how to flip $p(\mu,\sigma|D,I) \leftrightarrow p(D|\mu,\sigma,I)$. Here the first is hard but the second is easy.

:::{note}
Aside on the denominator, which is called in various contexts the evidence, the data probability, or the fully marginalized likelihood.
We evaluate it by using the basic marginalization rule (from the sum rule) to first insert an integration over all values of the general vector of parameters $\boldsymbol{\theta}$ and then the product rule to obtain an integral over the probability to get the data $D$ given a particular $\boldsymbol{\theta}$ times the probability of that $\boldsymbol{\theta}$:

$$ \begin{align}
 p(D|I) &= \int p(D,\boldsymbol{\theta}| I) \, d\boldsymbol{\theta} \\
    &= \int p(D|\boldsymbol{\theta}, I) p(\boldsymbol{\theta})\, d\boldsymbol{\theta}
\end{align} $$

This is numerically a costly integral.
Later we will look at ways to evaluate it.
:::

* For model *fitting* (i.e., parameter estimation), we don't need $p(D|I)$ calculated. Instead we find the posterior and directly normalize that function or, most often we only need relative probabilities.

* If $p(\mu,\sigma | I) \propto 1$, this is called a "flat" or "uniform" prior, in which case

$$
  p(\mu,\sigma | D,I) \propto \mathcal{L}(D|\mu,\sigma)
$$

and a Frequentist and Bayesian will get the same answer for the most likely values $\mu_0,\sigma_0$ (called "point estimates" as opposed to a full pdf).
    * We will argue against the use of uniform priors later.

* The prior includes additional knowledge (information). It is what you know *before* the measurement in question.

