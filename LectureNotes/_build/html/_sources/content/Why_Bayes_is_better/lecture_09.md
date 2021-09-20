# Lecture 9


## Why Bayes is Better I

* These examples were developed by Christian Forssén for the [2019 TALENT course at York, UK](https://nucleartalent.github.io/Bayes2019).
* Notebooks we'll use:
    * [](/notebooks/Why_Bayes_is_better/bayes_billiard.ipynb)
    * [](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb)
    * [](/notebooks/Why_Bayes_is_better/error_propagation_to_functions_of_uncertain_parameters.ipynb)

## Quotes from one pioneering and one renaissance Bayesian authority
> *"Probability theory is nothing but common sense reduced to calculation."*
(Laplace)

> *"Bayesian inference probabilities are a measure of our state of knowledge about nature, not a measure of nature itself."*
(Sivia)

## Summary: Advantages of the Bayesian approach

&nbsp;&nbsp;&nbsp;&nbsp; 1. Provides an elegantly simple and rational approach for answering, in an optimal way, any scientific question for a given state of information. This contrasts to the recipe or cookbook approach of conventional statistical analysis. The procedure is well-defined:
  - Clearly state your question and prior information.
  - Apply the sum and product rules. The starting point is always Bayes’ theorem.
  
For some problems, a Bayesian analysis may simply lead to a familiar statistic. Even in this situation it often provides a powerful new insight concerning the interpretation of the statistic.

&nbsp;&nbsp;&nbsp;&nbsp; 2. Incorporates relevant prior (e.g., known signal model or known theory model expansion) information through Bayes’ theorem. This is one of the great strengths of Bayesian analysis. 
  - For data with a small signal-to-noise ratio, a Bayesian analysis can frequently yield many orders of magnitude improvement in model parameter estimation, through the incorporation of relevant prior information about the signal model.
  - For effective field theories, information about the expansion can be explicitly included and tested.

&nbsp;&nbsp;&nbsp;&nbsp; 3. Provides a way of eliminating nuisance parameters through marginalization. For some problems, the marginalization can be performed analytically, permitting certain calculations to become computationally tractable.

&nbsp;&nbsp;&nbsp;&nbsp; 4. Provides a way for incorporating the effects of systematic errors arising from both the measurement operation and theoretical model predictions.

&nbsp;&nbsp;&nbsp;&nbsp; 5. Calculates probability of hypothesis directly: $p(H_i|D, I)$.

&nbsp;&nbsp;&nbsp;&nbsp; 6. Provides a more powerful way of assessing competing theories at the forefront of science by automatically quantifying Occam’s razor. 
 

### Occam's razor
Occam’s razor is a principle attributed to the medieval philosopher William of Occam (or Ockham). The principle states that one should not make more assumptions than the minimum needed. It underlies all scientific modeling and theory building. It cautions us to choose from a set of otherwise equivalent models of a given phenomenon the simplest one. In any given model, Occam’s razor helps us to "shave off" those variables that are not really needed to explain the phenomenon. It was previously thought to be only a qualitative principle. 

The Bayesian quantitative Occam’s razor can also save a lot of time that might otherwise be spent chasing noise artifacts that masquerade as possible detections of real phenomena.
We'll have much more to say about this later when we discuss the Bayesian evidence in detail!



## Nuisance parameters (I)

Nuisance parameters are parameters we introduce to characterize a situation but whih we don't care about or know in detail. We could also call them "auxiliary variables". The Bayesian way to deal with them is to marginalize, i.e., to integrate over them.

The procedure is illustrated in the notebook
["A Bayesian Billiard game"](/notebooks/Why_Bayes_is_better/bayes_billiard.ipynb)
and is quite generic, so it is worth looking at in detail. *The discussion here is not as complete as the notebook.*

Bayesian billiard schematic:
```{image} /_images/bayesian_billiard_schematic.png
:alt: Bayesian billiard schematic
:class: bg-primary
:width: 600px
:align: center
```
On a hidden billiard table (Alice and Bob can't see it), Carol has established $\alpha$, which is the fraction of the table defining winning positions for Alice and Bob. Alice wins a point if the balls ends up less than $\alpha$, otherwise Bob wins a point. The first to six wins. 

**Capsule summary:**  
* Carol knows $\alpha$ but Alice and Bob don't. 
* Alice and Bob are betting on various outcomes.
* After 8 rolls, the score is Alice 5 and Bob 3.
* They are now going to bet on Bob pulling out an overall win.
* Alice is most likely to win, she only needs 1 winning roll out of 3, and there is already some indication she is favored.
* **What odds should Bob accept?**

[Note: this is obviously not a physics problem but you can map it only many possible physic experimental or theoretical situations. E.g., $\alpha$ could be a normalization in an experiment (not between 0 and 1, but $\alpha_{\text{min}}$ and $\alpha_{\text{min}}$) or a model parameter in a theory that we don't know (we'll see examples later!).]

### Naive frequentist approach

Here we start by thinking about the best estimate for $\alpha$, call it $\alphahat$.
If $B$ is the statement "Bob wins," then what is $p(B)$?
* Bob winning a given roll has probability $1 - \alphahat$, and he must win 3 in a row $\Lra$ $p(B) = (1-\alphahat)^3$.
* For future reference: $p(B|\alpha) = (1-\alpha)^3$

Let's find the maximum likelihood estimate for $\alphahat$.
:::{admonition} What is the likelihood of $\alpha$ for the result Alice 5 and Bob 3?
:class: dropdown

$$
   \mathcal{L}(\alpha) = {8 \choose 5}\alpha^5 (1-\alpha)^3
$$

:::
:::{admonition} Given $\mathcal{L}(\alpha)$, find the maximum likelihood.
:class: dropdown

$$\begin{align}
   \Lra \left.\frac{\partial\mathcal{L}}{\partial\alpha}\right|_{\alphahat} =0
   & \Lra 5 \alphahat^4 (1 - \alphahat)^3 - 3 \alphahat^5 (1-\alphahat)^2 = 0 \\
   & \Lra 5(1-\alphahat) - 3\alphahat = 0 \\
   & \Lra \alphahat_{\text{MLE}} = 5/8 
\end{align}$$

:::
This estimate yields $p(B) \approx 0.053$ or 18 to 1 odds.

### Bayesian approach

You should try to fill in the details here!

:::{admonition}What pdf is the goal here?
:class: dropdown
Find $p(B|D,I)$ where $D = \{n_A = 5, n_B = 3\}$.
:::
:::{admonition} What would $I$ include here?
:class: dropdown
$I$ includes the details of the game.
:::
* Plan: introduce $\alpha$ as a nuisance parameter. If we know $\alpha$, the calculation is strightforward. If we only know it with some probability, then marginalize.
* Consider we can take several different equivalent paths to the same result.

$$\begin{align}
  &a.\ p(B|D,I) = \int_0^1 d\alpha\, p(B,\alpha|D,I)
    = \int_0^1 d\alpha\, p(B|\alpha,D,I) p(\alpha|D,I)\\
  &b.\ p(B,\alpha|D,I) \ \Lra\ \mbox{marginalize over $\alpha$}
    \ \Lra\ \mbox{back to a.} \\
  &c.\ p(B|\alpha,D,I) \ \Lra\ \mbox{marginalize, weighting by
  $p(\alpha|D,I)$}  
\end{align}$$

* What to do about $p(\alpha|D)$?
:::{admonition}What was the naive frequentist answer?
:class: dropdown
The naive frequentist used the MLE: $p(\alpha|D,I) = \delta(\alpha-\alphahat)$.
:::
The Bayesian approach is to use Bayes' theorem to write this pdf in terms of pdfs we know.
:::{admonition} Write it out
:class: dropdown

$$
 p(\alpha|D,I) = \frac{p(D|\alpha,I)p(\alpha|I)}{p(D|I)}
$$

:::

:::{admonition} What should we assume for the prior $p(\alpha|I)$?
:class: dropdown
The assumption is that there is no bias toward any value from 0 to 1, so we should assume a uniform pdf: $p(\alpha|I) = 1$ for $0 \leq \alpha \leq 1$ (with the implication that it is zero elsewhere).
:::

In this situation we will need the denominator (unlike other examples of Bayes' theorem we have considered) because we want a normalized probability.
:::{admonition} How do we evaluate the denominator?
:class: dropdown

$$
  p(D|I) = \int_0^1 d\alpha\, p(D|\alpha,I) p(\alpha|I)
$$

Note that we could write this directly or else first marginalize over $\alpha$ and then apply the product rule.
:::
Now put it all together:
:::{admonition} Find our goal!
:class: dropdown

$$\begin{align}
  p(B|D,I) &= \frac{\int_0^1 d\alpha\, p(B|\alpha,D,I) p(D,\alpha|I) p(\alpha|I)}
                  {\int_0^1 d\alpha\, p(D|\alpha,I) p(\alpha|I)} \\
           &= \frac{\int_0^1 d\alpha\, (1-\alpha)^3 {8\choose 5} \alpha^5 (1-\alpha)^3 \cdot 1}
                  {\int_0^1 d\alpha\, {8\choose 5} \alpha^5 (1-\alpha)^3 \cdot 1}
\end{align}$$

where $p(B|\alpha,D,I) = (1-\alpha)^3$ is just basic probability, $p(D|\alpha)$ follows from binomial probabilities, and note that the combinatoric factor canceled out in the end.

Can you directly interpret the first integral? It is an average of the probability of $B$ being true for a particular $\alpha$, weighted by the (normalized) probability of that $\alpha$.
:::

:::{admonition} What is the numerical result? Compare to the naive frequentist result.
:class: dropdown

$$ \Lra\ p(B|D,I) = \frac{int_0^1 d\alpha\, (1-\alpha)^6 \alpha^5}
          {int_0^1 d\alpha\, (1-\alpha)^3 \alpha^5}
          \approx 0.091
$$

or about 10 to 1 odds. Cf. 18 to 1 odds from our naive frequentist.
[Note: you can evaluate the integrals by expanding or by using the beta function $\beta(n,m) = \int_0^1 (1-t)^{n-1} t^{m-1}\, dt$.]
:::

So the predicted results are very different!  
:::{admonition} Why were the estimates so different? 
:class: dropdown
The frequentist evaluated the probability of Bob winning, $p(B|\alpha,D,I)$ at the peak value of the weighting probability (maximum likelihood estimate), while the Bayesian *integrated* over that pdf. Because the pdf is very broad and asymmetric, these gave quite different answers.
:::

:::{admonition} How do we check who is correct?
:class: dropdown
In many cases we can do a Monte Carlo simulation (at least to validate test cases). See the notebook [](/notebooks/Why_Bayes_is_better/bayes_billiard.ipynb) for an mplementation of this simulation. The result? Bayes wins!!!
::: 

Discussion points:
* Introducing $\alpha$ is straightforward in a Bayesian approach, and all assumptions are clear.
* In general one introduces *many* such variables, which is how we can end up with posterior integrals we need to sample to do marginalization.
* The problem with the "naive frequentist" approach is not that it is "frequentist" but that it is "naive". (In this case an incorrect use of a MLE to predict the likelihood of the result $B$.)
But it is not easy to see how to proceed to take into account the need to sum over possibilities for $\alpha$, while it is natural for Bayes. Bayes is better!

