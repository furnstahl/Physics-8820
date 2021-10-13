# Lecture 15

## Recap of evidence ratio



## Evidence for model EFT coefficients

* Problem: In the toy model example for model selection, the evidence is highly sensitive to the prior. One solution is to establish a clear criterion 

The notebook [](Evidence_for_model_EFT_coefficients.ipynb)



## Evidence with linear algebra

Return to the notebook to look at the calculation of evidence with Python linear altegra.

* The integrals to calculate are Gaussian in muliple variables: $\avec = (a_0, \ldots, a_k)$ plus $\abar$.

* We can write them with matrices (see [](/content/Why_Bayes_is_better/lecture_13.md) notes).

    $$
     \chi^2 = [Y - A\thetavec]^\intercal\, \Sigmavec^{-1}\, [Y -     A\thetavec]
    $$


Here we have a couple of options:

i) Use 

$$ 
  \int e^{-\frac{1}{2}x^\intercal M x + \Bvec^\intercal x} dx
= \sqrt{\det(2\pi M^{-1})} e^{\frac{1}{2}\Bvec^\intercal M^{-1} \Bvec}
$$ 

where $M$ is any square matrix and $\Bvec$ any vector. Derive this result by completing the square in the exponent (subtract and add $\frac{1}{2}\Bvec^\intercal M^{-1} \Bvec$).

ii) Use conjugacy. See the "[conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)" entry in Wikipedia for details. 
Apply this to Bayes' theorem with a Gaussian prior $p(\thetavec)$ and Gaussian likelihood $p(D|\thetavec)$.  
