# Lecture 10

We continue with more examples of why Bayes is better! 

## Nuisance parameters (II)

Here we return to the standard problem of fitting a straight line, this time for a real physics case: velocities (the $y$ variables) and distances (the $x$ variables) for a set of galaxies.
* A constant standard deviation of $\sigma = 200\,\mbox{km/sec}$ is given for the $y$ values and no error is given for $x$.
* The question: What value and error should we adopt for the Hubble constant (the slope), assuming we believe that a straight line is a valid model? Further, we don't care about the intercept; indeed, the model is $v = H_0 x$. 

We'll compare three estimates in this notebook:
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb)
The three approaches are:
1. Maximum likelihood estimate
1. Single-parameter inference
1. Full Bayesian analysis

## Error propagation: marginalization
[Finding the slope of a straight line (part II)](/notebooks/Why_Bayes_is_better/parameter_estimation_fitting_straight_line_II.ipynb#step-4-error-propagation)

## Error propagation: prior information
[Example: Propagation of systematic errors](/notebooks/Why_Bayes_is_better/error_propagation_to_functions_of_uncertain_parameters.ipynb)

