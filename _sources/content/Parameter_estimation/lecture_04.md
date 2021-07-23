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

