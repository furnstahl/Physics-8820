# Lecture 11

## Error propagation: functions of uncertain parameters

To propagate errors, we need to know the answer to:
Given a posterior for $X$, what is the posterior for $Y = f(X)$?


Here is a schematic of the posteriors for $X$ and $Y$ (the notation is that $x$ is an instance of the random variable denoted $X$):
```{image} /_images/functions_of_uncertain_parameters_handdrawn.png
:alt: p(x) and p(y) given Y = f(X) schematic
:class: bg-primary
:width: 600px
:align: center
```
We are assuming here that there is a 1-1 mapping of the function.

So what do we know? **The probability in the interval shown in each picture must be the same, regardless of what variable is used, $X$ or $Y$.** Note that this is the probability, not the probability density.
Therefore

$$
  p(X=x^* | I)\delta x = p(Y=y^* | I) \delta y
  \qquad \mbox{with}\ y^* = f(x^*)
$$

This must be true for all $x^*$, so in the $\delta x,\delta y \rightarrow 0$ limit we must have

$$
  p(x|I) = p(y|I)\times \left| \frac{dy}{dx} \right| .
$$

An alternative derivation uses marginalization and the fact that $y = f(x)$ means the pdf $p(y|x,I)$ is a delta function:

$$\begin{align}
  p(y^* | I) &= \int p(y^* | x,I) p(x|I) \, dx \\
     &= \int \delta\bigl(y^* - f(x)\bigr) p(x|I) \, dx \\
     & = \int \left| \frac{1}{df/dx}\right|_{x^*} p(x|I)\, dx \\
     & = \frac{1}{\left|df/dx\right|_{x^*}} p(x^* | I),
\end{align}$$

where we have used the properties of delta functions.
So it is really just a matter of changing variables.

As an example of how a more naive (but apparently faster) approach can fail badly, 
consider the example from Sivia 3.6.2 in the notebook:
[Example: Propagation of systematic errors](/notebooks/Why_Bayes_is_better/error_propagation_to_functions_of_uncertain_parameters.ipynb)


## Visualization of the Central Limit Theorem (CLT)

Go through the [Visualization of the Central Limit Theorem](/notebooks/Basics/visualization_of_CLT.ipynb) notebook.

### Follow-up to CLT proof for Poisson distribution

In an assignment you proved that in the limit $D \rightarrow \infty$ 
   
$$
\frac{D^N e^{-D}}{N!} \stackrel{D\rightarrow\infty}{\longrightarrow} \frac{1}{\sqrt{2\pi D}}e^{-(N-D)^2/(2D)}
$$

using Stirling's formula:  $x! \rightarrow \sqrt{2\pi x}e^{-x} x^x$ as $x\rightarrow\infty$ and letting $x = N = D(1+\delta)$ where $D\gg 1$ and $\delta \ll 1$,  plus $(1+\delta)^a = e^{a\ln (1+\delta)}$.

:::{admonition} In carrying out this proof, the term $D\delta^2/2$ is kept but $\delta/2$ is dropped. Why?
:class: dropdown
Because $N-D$ is of order the standard deviation of the distribution, which is $\sqrt{D}$. But $N-D = D\delta$,
so $D\delta \sim \sqrt{D} \gg 1$, which means $D\delta^2/2 = \delta/2 (D\delta) \gg \delta/2$.
:::


## PDF Manipulations I: Combining random variables