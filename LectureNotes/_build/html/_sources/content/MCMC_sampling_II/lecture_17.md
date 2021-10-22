# Lecture 17

## Parallel tempering summary points

* Simulate $N$ replicas of a system at different $\beta = 1/T$, where the temperature dependent log posterior is 

    $$
       log p_\beta(\thetavec|D,I) = C + \beta \log p(D|\thetavec,I) + \log p(\thetavec|I) .
    $$

    * The temperatures range from $\beta$ small ($T$ large) to $\beta = 1$, which is the results we are trying to find. The user chooses the $\beta$ values.
    * $\beta = 0$ samples the prior, so it is spread over the accessible parameter space.

* The $N$ chains run in parallel. A swap of configurations is proposed at random intervals between adjacent chains. A Metropolis-like criterion is used to decide whether the swap is selected or not.
* The evidence can be calculated numerically by thermodynamic integration using the results at all the temperatures in a numerical quadrature formula.

## Hamiltonian Monte Carlo (HMC)

* We've seen some different strategies for sampling difficult posteriors, such as an affine-invariant sampling approach (emcee) and a thermodynamic approach (parallel tempering).

* One of the most widespread techniques in contemporary samplers is Hamiltonian Monte Carlo, or HMC.
    * We'll look at some visualizations as motivation, then consider some examples using PyMC3.

* We return to the excellent set of interactive demos by Chi Feng at [https://chi-feng.github.io/mcmc-demo/](https://chi-feng.github.io/mcmc-demo/) and their adaptation by Richard McElreath at [http://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/](http://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/). These are also linked on the 8820 Carmen visualization page.