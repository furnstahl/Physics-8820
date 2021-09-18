# Lecture 8

## Recap of Poisson MCMC example

Notebook: [](/notebooks/MCMC_sampling_I/Metropolis_Poisson_example.ipynb)

:::{admonition} Recall Metropolis algorithm for $p(\thetavec | D, I)$ (or any other posterior).
* start with inital point $\thetavec_0$ (for each walker)
* **begin repeat:** given $\thetavec_i$, propose $\phivec$ from $q(\phivec|\thetavec_i)$
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
* **end repeat** when "converged".

Key questions: When are you converged? How many "warm-up" or "burn-in " steps to skip?
:::

**Poisson take-aways:**
1. It works! Sampled historgram agrees with (scaled) exact Poisson pdf (that is what success looks like). But not *normalized*! Compare 1000 to 100,000.
1. Warm-up time is (apparently) seen from the trace. *Moral: always check traces!*
1. The trace also shows that the space is being explored (rather than being confined to one region).
1. What if the $\thetavec_{i+1}=\thetavec_i$ step is not implemented as it should be? (I.e., so the chain is only incremented if the step is accepted.) This is not clearly intuitive. But see the figures below with 100,000 steps. The first one follows the Metropolis algorithm and adds the same step if the candidate is rejected; the second one does not. Not keeping the repeated steps invalidates the Markov chain conditions $\Lra$ wrong stationary distribution (not by a lot but noticeably and every time it is run).  
<br/>
<img src="/_images/MCMC_poisson_100000_with_repeats.png" alt="MCMC poisson results with repeats" class="bg-primary mb-1" width="450px"><img src="/_images/MCMC_poisson_100000_no_repeats.png" alt="MCMC poisson results with no repeats" class="bg-primary mb-1" width="450px">
1. The spread of the means decreases as $1/\sqrt{N_{\text{steps}}}$, as expected.

:::{admonition} Summary points from [arXiv:1710.06068](https://arxiv.org/abs/1710.06068)
* Before considering another example, some summary points from  ["Data Analysis Recipes: Using Markov Chain Monte Carlo"](https://arxiv.org/abs/1710.06068) by David Hogg and Daniel Foreman-Mackey.
    * Both are computational astrophysicists (or cosmologists or astronomers)
    * DFM wrote `emcee`.
    * Highly experienced in physics scenarios, highly opinionated, but not statisticians (although they interact with statisticians).

* MCMC is good for *sampling*, but not optimizing. If you want to find the modes of distributions, use an optimizer instead.
* For MCMC, you only have to calculate *ratios* of pdfs (as seen from the algorithm).
     * $\Lra$ don't need analytic normalized pdfs
     * $\Lra$ great for sampling posterior pdfs

$$
  p(\thetavec|D) = \frac{1}{Z} p(D|\thetavec)p(\thetavec)
$$     

* Getting $Z$ is really difficult because you need *global* information. (Cf. $Z$ and partition function in statistical mechanics.)

* MCMC is extremely easy to implement, without requiring derivatives or integrals of the function (but see later discussion of HMC).

* Success means a histogram of the samples looks like the pdf.

* Sampling for expectation values works even though we don't know $Z$; we just need the set of $\{\thetavec_k\}$.

$$\begin{align}
  E_{p(\thetavec)}[\thetavec] &\approx \frac{1}{N}\sum_{k=1}^{N}\thetavec_k \\
  E_{p(\thetavec)}[g(\thetavec)] &\approx \frac{1}{N}\sum_{k=1}^{N}g(\thetavec_k)
  \longrightarrow
  \frac{\int d\thetavec\, g(\thetavec)f(\thetavec)}{\int d\thetavec\, f(\thetavec)}
\end{align}$$

where $f(\thetavec) = Z p(\thetavec)$ is unnormalized.

* Nuisance parameters are very easy to marginalize over: just drop that column in every $\theta_k$.

* Autocorrelation is important to monitor and one can tune (e.g., Metropolis step size) to minimize it. More on this later.

* How do you know when to stop? Heuristics and diagnostics to come!
* Practical advice for initialization and burn-in is given by Hogg and Foreman-Mackey.
:::

## MCMC random walk and sampling example

Look back at the notebook [](/notebooks/MCMC_sampling_I/MCMC-random-walk-and-sampling.ipynb). Let's do some of the first part together.

**Part 1:** Random walk in the $[-5,5]$ region. The proposal step is drawn from a normal distribution with zero mean and standard deviation `proposal_width`.

Algorithm: always accept unless across boundary.

Class: map this problem onto the Metropolis algorithm.
* What is $q(\phivec|\thetavec_i)$? $\Lra$
   $\phi \sim \mathcal{N}(\theta_i,(\text{proposal-width})^2)$
:::{hint}
Use `shift-tab-tab` to check whether the normal function takes $\sigma$ or $\sigma^2$ as an argument.
:::
:::{admonition} What is $p(\thetavec_i|D,I)$?
:class: dropdown
It must be constant except for the borders $\Lra$ $U(-5,5)$
:::
* Check the answer. Change `np.random.seed(10)` to `np.random.seed()`
    * This will mean a different pseudo-random number sequence every time you re-run.
    * Note the fluctuations.
    * Try 200 then 2000 samples.
    * Try changing to a uniform proposal.
    * Try not adding the rejection position.
    :::{hint}
    :class: dropdown
    Move `samples.append(current_position)` under the `if` statement.
    Result: it doesn't fill the full space (try different runs -- trouble with edges)
    :::
    * Decrease width to 0.2, do 2000 steps $\Lra$ samples are too correlated.
    * Look at definition and correlation time for 0.2 and 2.0.
    * Trend in autocorrelation plot: 1 at lag $h=0$; how fast to fluctuations around 0?


