# Lecture 18

## What is expected for your project?

In most cases, you will turn in a Jupyter notebook relevant to your research interests (which doesn't has to be your thesis research area!) using Bayesian tools. As already stressed in class, the project does not have to be completed if it will continue as part of your thesis work.

## Overview of Intro to PyMC3 notebook

[](/notebooks/MCMC_sampling_II/PyMC3_intro_updated.ipynb) starts with sampling to find the posterior for $\mu$, the mean of a distribution, given data that is generated according to a normal distribution with mean zero.
* Try changing the true $\mu$, sampling sigma as well.
* The standard deviation is initially fixed at 1.

In the code, one specifies priors and then the likelihood. This is sufficient to specify the posterior for $\mu$ by Bayes' theorem (up to a normalization):

$$
  p(\mu|D,\sigma) \propto p(D|\mu,\sigma) p(\mu | \mu_\mu^0, \sigma_\mu^0) ,
$$
    
* $p(\mu|D,\sigma)$ is the pdf to sample,
* $D$ is the *observed* data,
* $p(D|\mu,\sigma)$ is the likelihood, which is specified last. Here it is $\sim \mathcal{N}(\mu,\sigma^2)$.
* $\mu_\mu^0$ and $\sigma_\mu^0$ are hyperparameters specifying the priors for $\mu$ and $\sigma$. The statements for these priors appear first.

## Looking at PyMC3 getting started notebook

[](/notebooks/MCMC_sampling_II/PyMC3_docs_getting_started_updated.ipynb) starts with a brief overview and a list of features of PyMC3. Note the comment about Theano being used to calculate gradients needed for HMC. The technique of ["automatic differentiation"](https://en.wikipedia.org/wiki/Automatic_differentiation) ensures machine precision results for these derivatives.

Consider the "Motivating Example" on linear regression model that predicts the outcome $Y \sim \mathcal{N}(\mu,\sigma^2)$ from a "linear model", meaning the expected result is a linear combination of the two input variables $X_1$ and $X_2$, $\mu = \alpha + \beta_1 X_1 + \beta_2 X_2$. 
* This is what we have written elsewhere as 

    $$
      y_{\text{expt}} = y_{\text{th}} + \delta y_{\text{exp}}
    $$

    (no model discrepancy $\delta y_{\text{th}}$), with $y_{\text{expt}} \rightarrow Y$, $y_{\text{th}} \rightarrow \mu$, and $\delta y_{\text{exp}}$ Gaussian noise with mean zero and variance $\sigma^2$.
* The basic workflow is: make data then encode the model in approximate statistical notation.
* Step through the line-by-line explanation of the model (more detail than here!). 
    * Use the Python `with` for a context manager. Note that the first two statements could have been combined to be `with pm.Model() as basic_model:`.
    * Compare the code to first the priors for *unknown* parameters:

        $$\begin{align}
          \alpha \sim \mathcal{N}(0,100), \quad 
          \beta_i \sim \mathcal{N}(0,100), \quad
          \sigma \sim |\mathcal{N}(0,1)|,
        \end{align}$$

        Investigate the options for `Normal`, such as `shape` in the documentation for [Probability Distributions in PyMC3](https://docs.pymc.io/en/stable/Probability_Distributions.html).

    * Then the encoding of $\mu = \alpha + \beta_1 X_1 + \beta_2 X_2$.

    * Finally the statement of the likelihood (always comes last) $Y \sim \mathcal{N}(\mu,\sigma^2)$. Note the use of `observed=Y`.
    * Read through all the background!. This and the other examples can serve as prototypes for your use of PyMC3.


Your task: run through the notebook!
