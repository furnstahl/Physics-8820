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

## Review: Advantages of the Bayesian approach

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
 
The Bayesian quantitative Occam’s razor can also save a lot of time that might otherwise be spent chasing noise artifacts that masquerade as possible detections of real phenomena. 

### Occam's razor
Occam’s razor is a principle attributed to the medieval philosopher William of Occam (or Ockham). The principle states that one should not make more assumptions than the minimum needed. It underlies all scientific modeling and theory building. It cautions us to choose from a set of otherwise equivalent models of a given phenomenon the simplest one. In any given model, Occam’s razor helps us to "shave off" those variables that are not really needed to explain the phenomenon. It was previously thought to be only a qualitative principle. 
![leprechaun](https://upload.wikimedia.org/wikipedia/commons/5/55/Leprechaun_or_Clurichaun.png)

## Nuisance parameters (I)
[A Bayesian Billiard game](/notebooks/Why_Bayes_is_better/bayes_billiard.ipynb)

## Nuisance parameters (II)
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb)

## Error propagation: marginalization
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb#step-4-error-propagation)

## Error propagation: prior information
[Example: Propagation of systematic errors](/notebooks/Why_Bayes_is_better/error_propagation_to_functions_of_uncertain_parameters.ipynb)

