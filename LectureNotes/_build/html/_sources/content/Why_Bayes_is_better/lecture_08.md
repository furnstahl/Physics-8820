# Lecture 8

## Recap of Poisson MCMC example

Metropolis_Poisson_example.ipynb

:::{admonition} Recall Metropolis algorithm for this example and the next.
* start with $\thetavec_0$
* repeat: given $\thetavec_i$, propose $\phivec$ from $q(\phivec|\thetavec_i)$
* calculate $r$:

$$
  r = \frac{p(\phivec|D,I)}{p(\thetavec_i|D,I)}
    \left[\frac{q(\thetavec_i|\phivec)}{q(\phivec|\thetavec_i)}\right]
$$

where the left factor compares posteriors and the right factor is a correction for $r$ if the $q$ pdf is not symmetric: $q(\phivec|\thetavec_i) \neq q(\thetavec_i|\phivec)$
* decide whether to keep $\phivec$:
    * if $r\geq 1$, set $\thetavec_{i+1} = \phivec$ (accept)
    * else draw $u \sim \text{uniform}(0,1)$
        * if $u \leq r$, $\thetavec_{i+1} = \phivec$ (accept)
        * else $\thetavec_{i+1} = \thetavec_i$ (add another copy of $\thetavec_i$ to the chain)
* repeat until "converged".

Key questions: When are you converged? How many "warm-up" or "burn-in " steps to skip?
:::

**Poisson take-aways:**
1. It works! Sampled historgram agrees with (scaled) exact Poisson pdf (that is what success looks like). But not *normalized*! Compe 1000 to 100,000.
1. Warm-up time is (apparently) seen from the trace. *Moral: always check traces!*
1. The trace also shows that the space is being explored.
1. What if the $\thetavec_{i+1}=\thetavec_i$ step is not implemented as it should be? (I.e., so the chain is only incremented if the step is accepted.) This is not clearly intuitive. See Metropolis_Poisson_example_with_results_no_repeats.ipynb $\Lra$ compare 100,000 $\Lra$ invalidates Markov chain conditions $\Lra$ wrong stationary distribution.
1. The spread of the means descrease as $1/\sqrt{N_{\text{steps}}}$, as expected.

:::{admonition} Summary points from [arXiv:1710.06068](https://arxiv.org/abs/1710.06068)
* Before considering another example, some summary points from  "Data Analysis Recipes: Using Markov Chain Monte Carlo" by David Hogg and Daniel Foreman-Mackey.
    * Both computational astrophysicists (or comoslogists or astronomers)
    * DFM wrote `emcee`.
    * Highly experienced in physics scenarios, highly opinionated, but not statisticians (although they interact with statisticians).

* MCMC is good for *sampling*, but not optimizing. If you want to find modes of distributions, use an optimizer.
* For MCMC, you only have to calculate *ratios* of pdfs
     * $\Lra$ don't need analytic normalized pdf
     * $\Lra$ great for sampling posterior pdfs

$$
  p(\thetavec|D) = \frac{1}{Z} p(D|\thetavec)p(\thetavec)
$$     

* Getting $Z$ is really difficult because you need *global* information.

* MCMC is extremely easy to implement, without requiring derivatives or integrals of the function (but see later discussion of HMC).

* Success means a histogram of the samples looks like the pdf.

* Sampling for expectation values works even though we don't know $Z$.

$$\begin{align}
  E_{p(\thetavec)}[\thetavec] &\approx \frac{1}{N}\sum_{k=1}^{N}\thetavec_k \\
  E_{p(\thetavec)}[g(\thetavec)] &\approx \frac{1}{N}\sum_{k=1}^{N}g(\thetavec_k)
  \longrightarrow
  \frac{\int d\thetavec\, g(\thetavec)f(\thetavec)}{\int d\thetavec\, f(\thetavec)}
\end{align}$$

where $f(\thetavec) = Z p(\thetavec)$ is unnormalized.

* Nuisance parameters are very easy to marginalize: just drop that column.

:::


