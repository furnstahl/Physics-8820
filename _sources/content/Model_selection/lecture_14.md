# Lecture 14

## Bayesian model selection (or model comparison)

* The discussion here is based heavily on Sivia, Chapter 4 {cite}`sivia2006`.

* We've mostly focused so far on parameter estimatioin: *given* a model with parameters, what is the joint posterior for those parameters given some data. This is what we Bayesians mean by fitting the parameters, finding: $p(\thetavec| D,I)$.

* Now we turn to an analysis of the model itself, or, more precisely, to the *comparison* of models.
    * Remember this: model selection will always be about comparisons.
    * We can think of many possible situations (all given data):
        * Is the signal a Gaussian or Lorentzian line shape?
        * When fitting to data, what order polynomial is best?
        * Given two types of Hamiltonian, which is better?
        * In an effective field theory (EFT), which *power counting* (organization of Feynman diagrams) is favored?
    * In all cases, we are not asking about the best fit but which model is most suitable.
    * Note that if you consider the polynomial case, if you decided by how well polynomials fit the data, e.g., by finding the least residual, then higher order will *always* be better (or, at least equal, as you can set coefficients to zero as a special case).

* So let's think how a Bayesian would proceed. As is usually the best strategy, we'll start with the simplest possible example, dating back to Jeffreys (1939) $\longrightarrow$ Gull (1988) $\longrightarrow$ Sivia (2006) $\longrightarrow$ TALENT (2019):
The story of Dr. A and Prof. B. The basis setup is:

>"Dr. A has a theory with no adjustable parameters. Prof. B also has a theory, but with an adjustable parameter $\lambda$. Whose theory should we prefer on the basis of data $D$?"

* For example, this could be data $D$: &nbsp;&nbsp;<img src="/_images/data_for_Dr_A_and_Prof_B_handdrawn.png" alt="schematic data for Dr. A and Prof. B" class="bg-primary mb-1" width="200px"> <br/>
Dr. A thinks $y\equiv 0$. Prof. B thinks $y=\lambda$, with $\lambda$ to be determined.

* Our Bayesian approach is to consider a ratio of posteriors:

    $$
     \frac{p(A|D,I)}{p(B|D,I)}
    $$

    * We emphasize: the denominator is not the probability density for a particular instance of $B$, such as $\lambda =0.2$, but the pdf that the theory is correct. $\lambda$ doesn't appear yet.
