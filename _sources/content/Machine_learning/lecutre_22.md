# Lecture 22

## MaxEnt function reconstruction

Quick summary of [](/notebooks/Maximum_entropy/MaxEnt_Function_Reconstruction):

* Consider a function on $x \in [0,1]$ with $f(x)>0$.
Suppose we know *moments* of the function:

    $$
      \mu_j = \int_0^1 dx \, x^j f(x), \quad j=0,\ldots,N
    $$

    Our goal is to reconstruct $f(x)$ from the $N$ moments.

* Define the functional

    $$
    S[f] = -\int_0^1 dx\, \bigl(f(x)\log f(x) - f(x) \bigr)
    $$

* Maximize $S$ subject to the constraints given by the moments 

    $\Lra$ this gives $f(x)$.

* Mead and Papanicolaou formulated this; it is implemented in the notebook.


## Bayesian methods in machine learning (ML): background remarks

* ML encompasses a broad array of techniques, many of which do not require a Bayesian approach, or they may even have a philosophy largely counter to the Bayes way of doing things.
    * But there are clearly places where Bayesian methods are useful.
    * We will tough upon two examples:
        1. Bayesian Optimization
        2. Bayesian Neural Networks
    * Given time limitations, these will necessarily only be teasers for a more complete treatment.

* Let's step through the Neal_Bayesian_Methods_for_Machine_Learning_selected_slides. Some comments:
    1. These came from 2004, which seems out-of-date, but the underlying ML ideas have been around for a long time. Recent successes have stemmed from refinements of old ideas (some of which were thought not to work, but just needed implementation tweaks).
    2. Bayesian Approach to ML (or anything!).
        * Emphasis that the approach is very general.
        * Some of the uses in step 4) are *often* different in ML.
        * Often in ML one doesn't account for uncertainty, and optimize to find predictions. But some applications require an assessment of risk (e.g., medical applications) or a non-black-box idea of how the conclusion was reached (e.g., legal applications).
    3. Distinctive features set up to contrast with black-box ML.
    4. "Learning Machine" approach is *one way* to view ML. 
        * Note that is works best with "big data", i.e., a log of data, in which case we know that Bayesian priors become less important.
        * Conversely, the relevant of a Bayesian approach is greater when data is limited (or expensive to get).  
