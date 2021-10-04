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

### Adding independent random variables

* Suppose we have two random variables, $X$ and $Y$, drawn from two different distributions.
    * For concreteness we'll take the distributions to be Gaussian (normal) with different means and widths, and *independent* of each other.

    $$\begin{align}
      X | \mu_x,\sigma_x &\sim \mathcal{N}(\mu_x,\sigma_x^2)
       \quad\Longleftrightarrow\quad
       p(x|\mu_x,\sigma_x) = \frac{1}{\sqrt{2\pi}\sigma_x}e^{-(x-\mu_x)^2/2\sigma_x^2} \\
      Y | \mu_y,\sigma_y &\sim \mathcal{N}(\mu_y,\sigma_y^2)
       \quad\Longleftrightarrow\quad
       p(y|\mu_y,\sigma_y) = \frac{1}{\sqrt{2\pi}\sigma_y}e^{-(y-\mu_y)^2/2\sigma_y^2} 
    \end{align}$$

* What is the distribution of the *sum* of $X$ and $Y$?
    * From past experience, we might expect that "errors add in quadrature", but how does that arise in detail?
    * This is a special case of *uncertainty propagation*: How do errors combine?

    
    :::{admonition} First phrase the question in terms of a statement     about posteriors
    :class: dropdown
    Goal: given $Z=X+Y$, how is $Z$ distributed, i.e., what is $p(z|I)$?
    :::

* Plan: follow the same steps as with the CLT proof! (We'll suppress the explicit information $I$'s here for conciseness.)

    $$\begin{align}
      p(z) &= \int_{-\infty}^{\infty} dx\,dy\, p(z,x,y) \\
           &= \int_{-\infty}^{\infty} dx\,dy\, p(z|x,y) p(x,y) \\
           &= \int_{-\infty}^{\infty} dx\,dy\, p(z|x,y) p(x) p(y)
    \end{align}$$ (add_1)

    :::{admonition} What is the justification for each of the three steps?
    :class: dropdown
    Marginalization, product rule, independence. What if $X$ and $Y$ are not independent?
    :::


    :::{admonition} What is $p(z|x,y)$?
    :class: dropdown
     
    $$
      p(z|x,y) = \delta(z-x-y) =
        \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega\,
        e^{-i\omega\left(z-(x+y)\right)}
        = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega\,
        e^{-i\omega z} e^{i\omega x} e^{i\omega y}
    $$ (add_2)

    We choose the Fourier representation of the delta function because the dependence on $x$ and $y$ *factorizes*. That is, the dependence appears as a product of a function of $x$ times a function of $y$. (But we don't need to do that for Gaussians; we can just use the delta function to do one integral and evaluate the second one directly.)

    :::

* Now we can substitute the result for $p(z|x,y)$ into the final expression for $p(z)$ and observe that the integrals *factorize*. Note: at this stage we have used independence of $X$ and $Y$ but we haven't used that they are Gaussian distributed.
So we get:

    $$\begin{align}
      p(z) &= \int_{-\infty}^{\infty} dx \int_{-\infty}^{\infty} dy
      \left(\int_{-\infty}^{\infty} \frac{d\omega}{2\pi} e^{-i\omega z}\right)
      e^{i\omega x} p(x)\, e^{i\omega y} p(y) \\
      &= \int_{-\infty}^{\infty} \frac{d\omega}{2\pi} e^{-i\omega z}
      \left[\int_{-\infty}^{\infty} dx\, e^{i\omega x}p(x)\right]
      \left[\int_{-\infty}^{\infty} dy\, e^{i\omega y}p(y)\right]
    \end{align}$$ (add_3)

* [Note: we really should verify that the functions are such that we can change integration orders to get the last line!]
    
* We now need the Fourier transforms of the pdfs for $X$ and $Y$. Finally we'll get specific and use Gaussians:

    $$\begin{align}
       \int_{-\infty}^{\infty} dx\, e^{i\omega x}
       \frac{1}{\sqrt{2\pi}\sigma_x}e^{-(x-\mu_x)^2/2\sigma_x^2}
       & = e^{i\mu_x\omega}e^{-\sigma_x^2\omega^2/2} \\
       \int_{-\infty}^{\infty} dy\, e^{i\omega y}
       \frac{1}{\sqrt{2\pi}\sigma_y}e^{-(y-\mu_y)^2/2\sigma_y^2}
       & = e^{i\mu_y\omega}e^{-\sigma_y^2\omega^2/2} \\
    \end{align}$$ (add_4)

* Here we have used $\int_{-\infty}^{\infty} e^{\pm iax} e^{-bx^2}dx = \sqrt{\frac{\pi}{b}} e^{-a^2/4b}$.

* Next substitute the FT's into the expression for $p(z)$:

    $$\begin{align}
      p(z) &= \int_{-\infty}^{\infty} \frac{d\omega}{2\pi} 
      e^{-i\omega (z - \mu_x - \mu_y)}
      e^{-\omega^2(\sigma_x^2 + \sigma_y^2)/2} \\
      &= \frac{1}{2\pi} \sqrt{\frac{2\pi}{\sigma_x^2 + \sigma_y^2}}
      e^{-(z - (\mu_x + \mu_y))^2/2(\sigma_x^2 + \sigma_y^2)} \\
      &= \frac{1}{\sqrt{2\pi}\sqrt{\sigma_x^2 + \sigma_y^2}}
      e^{-(z - (\mu_x + \mu_y))^2/2(\sigma_x^2 + \sigma_y^2)} 
    \end{align}$$ (add_5)

* Thus: $X+Y \sim \mathcal{N}(\mu_x+\mu_y, \sigma_x^2 + \sigma_y^2)$ or the means add and the *variances* add (not the standard deviations). 

* Generalizations:
    * $X-Y$ $\Lra$ Now $p(z|x,y) = \delta(z-(x-y))$ and the minus sign here takes $e^{i\omega y} \rightarrow e^{-i\omega y}$ and the FT yields $e^{-i\mu_y\omega}$ $\Lra$ $\mathcal{N}(\mu_x - \mu_y, \sigma_x^2 + \sigma_y^2)$.
    * $aX + bY$ for any $a,b$ $\Lra$ Now $p(z|x,y) = \delta(z-(ax+by))$ which yields $e^{-i\omega z}e^{ia\omega x}e^{ib\omega y}$. We show an example of how this means $\mu_x \rightarrow
    a\mu_x$ and $\sigma_x^2 \rightarrow a^2\sigma_x^2$ and similarly with $b$, leading to

    $$\begin{align}
     \int_{-\infty}^{\infty}dx\, e^{ia\omega x}
     \frac{1}{\sqrt{2\pi}\sigma_x}e^{-(x-\mu_x)^2/2\sigma_x^2}
     &= e^{ia\mu_x\omega}e^{-a^2\sigma_x^2\omega^2} \\
     &\Lra aX + bY \sim \mathcal{N}(a\mu_x+b\mu_y, a^2\sigma_x^2+b^2\sigma_y^2) .
    \end{align}$$

    * Note that this last example includes $X\pm Y$ as special cases.

    * Finally, we can sum $m$ Gaussian random variables, following all the same steps, to find

    $$
      X_1 + X_2 + \cdots + X_m \sim \mathcal{N}(\mu_1 + \mu_2 + \cdots+\mu_m, \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_m^2)
      .
    $$

    * Note that if we are given the FT's of the pdfs being combined, we can generalize {eq}`add_4` and {eq}`add_5` from the case of normal distributions.

### Adding correlated random variables

* Now suppose that $X$ and $Y$ are *jointly* normally distributed random variables, but not independent. So $p(x,y) \neq p(x)\cdot p(y)$.
* The general form of the joint normal distribution is 

    $$
      p(x,y) \equiv p(\xvec) =
      \frac{1}{\sqrt{(2\pi)^2 |\Sigmavec|}}
      e^{-\frac12 (\xvec - \muvec)^{\intercal} \Sigmavec^{-1}(\xvec-\muvec)}
      \quad\mbox{where}\
      \xvec = \pmatrix{x \\ y}
      \ \mbox{and}\ 
      \muvec = \pmatrix{\mu_x \\ \mu_y} .
    $$

    $|\Sigmavec|$ is the determininant of the $2\times 2$ symmetric, positive-definite matrix $\Sigmavec$ (i.e., all the eigenvalues are positive), called the "covariance matrix", which we parametetrize generally as

    $$
      \Sigmavec = \pmatrix{\sigma_x^2 & \rho\sigma_x\sigma_y \\
                           \rho\sigma_x\sigma_y & \sigma_y^2 
                           },
        \qquad\mbox{with}\ {-1} \leq \rho \leq {+1} .
    $$

    If $\rho = 0$, the problem is reduced to the uncorrelated case already considered.
* Returning to the derivation for adding independent $X$ and $Y$, how can we redo it from {eq}`add_1` now that $X$ and $Y$ are not independent?
One way is to skip the Fourier transform and evaluate the delta function, leaving a convolution of Gaussians. This can be evaluated directly (e.g., using Mathematica).
This is left as an exercise for the reader.
The result for the more general case is

    $$
       aX + bY \sim \mathcal{N}(a\mu_x + b\mu_y, 
       a^2\sigma_x^2 + b^2\sigma_y^2 + 2ab\rho\sigma_x \sigma_y).
    $$

* So if $\rho \approx 1$, then $\sim \mathcal{N}(a\mu_x + b\mu_y,
      (a\sigma_x + b\sigma_y)^2)$.
* Note that if $\rho < 0$ with $a\cdot b > 0$ or if $\rho > 0$ with $a\cdot b < 0$, then the variance is *reduced* compared to adding in quadrature.

* Consider the special case $X-Y$ with equal variances (for convenience):

    $$
       X - Y \sim \mathcal{N}\bigl(\mu_x-\mu_y, 2\sigma^2(1-\rho)\bigr).
    $$

    * If $\rho = 0$, we're back to adding in quadrature.
    * But if $\rho\approx 1$, then the error bar is reduced by a
    factor $\sqrt{1-\rho}$. (If $\sigma_x \gg \sigma_y$, it is still dominated by $\sigma_x^2$ and the correlation doesn't affect the result.)

### Last generalization: adding multivariate normals

* Now suppose $X$ and $Y$ are themselves random variables from (independent) multivariate normal distributions of dimension $N$ while $A,B$ are known $M\times N$ matrices. So

    $$
      X \sim \mathcal{N}(\muvec_x, \Sigmavec_x),
      \qquad
      Y \sim \mathcal{N}(\muvec_y, \Sigmavec_y),
    $$

    where $\muvec_x, \muvec_y$ are $N$-dimensional vectors
    and $\Sigmavec_x,\Sigmavec_y$ are $N\times N$ covariance matrices.

* The claim is that:

    $$
      A X + B Y \sim \mathcal{N}(A\muvec_x + B\muvec_y,
        A\Sigmavec_x A^\intercal + B\Sigmavec_y B^\intercal) .
    $$ 

:::{admonition} Check that the matrix-vector multiplications work
:class: dropdown
  $A\muvec_x \Lra (M\times N)\cdot N$ gives $M$ correctly;  $A\Sigmavec_x A^\intercal \Lra (M\times N)\cdot(N\times N)\cdot(N\times M)$ gives $M\times M$ correctly.        
:::

* If we apply this for $A \rightarrow a I_N$ and $B \rightarrow b I_N$, where $a,b$ are scalars and $I_N$ is the $N\times N$ identity, then it is much simpler:

    $$
      a X + b Y \sim \mathcal{N}(a\muvec_x + b\muvec_y, a^2\Sigmavec_x + b^2 \Sigmavec_y).
    $$

    So the *covariance* matrices add in quadrature!



