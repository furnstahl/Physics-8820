# Lecture 12

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
