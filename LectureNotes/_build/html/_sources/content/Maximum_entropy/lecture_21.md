# Lecture 21

## Follow-up to Gaussian process exercises

Return to the notebook [](/notebooks/Gaussian_processes/Gaussian_processes_exercises).

* Do sampling of different covariant functions in *2 Sampling from a Gaussian Process*. 
    * Predict `nsamples = 50`.

* Try some combinations
    * linear $\longrightarrow$ polynomial (try two to get quadratic). [See kernels.pdf Fig 1.1 (on page 2) and Fig. 1.2 (on page 4).]

* For Gaussian Process Regression Model
    * distinguish between noise in *data* and noise in *model*
    * compare the true function in red to the curves

* Other things you might play with:
    * Add a function for the true result (no noise) and add it (in red) to the plots.
    * Compare small data noise vs. large data noise. 
    * Try making lengthscale very small $\Lra$ explain the result. [Should return to the mean after a few length scales.]
    * Try optimizing with lengthscale very small (it doesn't change $\Lra$ optimize fails).    
    * With a good optimization, explore how well red line is withing the 95% bands.
        * relation to prior (mean zero, input variance)
        * What if I extend the range of `Xtrue`?

## Maximum entropy

A good reference here is Chapter 5 in Sivia {cite}`Sivia2006`.
The plan is to step through [](/notebooks/Maximum_entropy/MaxEnt), then [](/notebooks/Maximum_entropy/Pdfs_from_MaxEnt) as a class exercise. As time permits, we'll do [](/notebooks/Maximum_entropy/MaxEnt_Function_Reconstruction).

Notes on [](/notebooks/Maximum_entropy/MaxEnt):
* Ignorance pdfs $\longrightarrow$ when we don't have constraints or extra knowledge that breaks symmetries.
    1. permutation symmetry $\longrightarrow$ die $\Lra$
    1/(# choices) [discrete]
    2. translational invariance $\longrightarrow$ $p(x|I) = \text{constant}$ (in allowed region)
    3. scale invariance $\longrightarrow$ $p(x|I) \propto 1/x$
        * How to derive? 
        * First check that it works: $p(x|I) = \lambda p(\lambda x|I)$ $\Lra$ $\frac{c}{x} = \frac{\lambda c}{\lambda x} = \frac{c}{x}$. Check!
        * Now more general proof: assume $p(x|I) \propto x^{\alpha}$
        $\Lra$ $x^\alpha = \lambda (\lambda x)^{\alpha}
        = \lambda^{1+\alpha} x^\alpha$ $\Lra$ $\alpha = -1$. Check!
        * Still more general: set $\lambda = 1 + \epsilon$ with $\epsilon \ll 1$, and solve to $\mathcal{O}(\epsilon)$: $p(x) = (1+\epsilon)(p(x)+\epsilon\frac{dp}{dx})$ $\Lra$ $p(x) + x \frac{dp}{dx} = 0$

        $$
         \Lra \int_{p(x_0)}^{p(x)} \frac{dp}{p}
         = \int_{x_0}^x \frac{dx'}{x'}
         \ \Lra\ 
         \log\frac{p(x)}{p(x_0)} = \log\frac{x_0}{x}
         \ \Lra\  p(x) = \left(\frac{p(x_0)}{x_0}\right)\frac{1}{x}
        $$

        so $p(x) \propto 1/x$.

* Step quickly through Symmetry invariance.
    * Basically using a change of variables for the symmetry, which means a Jacobian. 
    * For the linear model: $y_{\rm th}(x) = \theta_1 x + \theta_0$, which we could write the other way around: $x_{\rm th}(y) = \theta'_1 y + \theta'_0$, and these probabilities (not densities!) should be equal:

    $$
     p(\theta_0,\theta_1|I) d\theta_0 d\theta_1
       = p(\theta'_0,\theta'_1|I) d\theta'_0 d\theta'_1
    $$

    * We can relate the primed and unprimed $\theta$'s by substituting:

    $$
    y = \theta_1 x + \theta_0 = \theta_1(\theta'_1 y + \theta'_0) + \theta_0
    = \theta_1 \theta'_1 y + \theta_1\theta'_0 + \theta_0
    $$

    $$
   \Lra \theta_1 \theta'_1 =1, \quad \theta_1\theta'_0+\theta_0 = 0
   \quad\Lra\quad \theta_1' = \theta_1^{-1}, \quad \theta'_0 = -\theta_1^{-1}\theta_0
    $$

    * This lets us calculate the Jacobian:

    $$
     \left| \det\pmatrix{ 
     \frac{\partial\theta_0}{\partial\theta'_{0}} &
     \frac{\partial\theta_0}{\partial\theta'_{1}} \\
     \frac{\partial\theta_1}{\partial\theta'_{0}} &
     \frac{\partial\theta_1}{\partial\theta'_{1}}}
     \right|
     =
     \left| 
     \begin{array}{cc}
       -\theta'_1{}^{-1} & \theta'_0 \\ 
     0 & -\theta'_1{}^{-2}
     \end{array} 
     \right|
     = \frac{1}{\theta'_1{}^3} = \theta_1^3 .
    $$

    * That means that 

    $$\begin{align}
     p(\theta_0,\theta_1|I) d\theta_0 d\theta_1
     & = p(-\theta_1^{-1}\theta_0,\theta_1^{-1}|I)  d\theta_0 d\theta_1 \frac{1}{\theta_1^3} \\
     \mbox{or}\ \theta_1^3 p(\theta_0,\theta_1|I)  &= p(-\theta_1^{-1}\theta_0,\theta_1^{-1}|I).
    \end{align}$$

    * One possible solution is

    $$
      p(\theta_0,\theta_1|I) \propto (1+\theta_1^2)^{-3/2} .
    $$
### Principle of Maximum Entropy

* Arguing from monkeys distributing $N$ balls into $M$ boxes, so $n_i$ in each box and $N = \sum_{i=1}^M n_i$.

* We'll let them do this many times, subject to the constraints described by $I$.
    * The idea is to find the pdf specified by $p_i = n_i/N$ for all $i$ that appears most often $\Lra$ this best represents our state of knowledge.

* So this becomes a matter of counting microstates (i.e., a particular distribution $\{n_i\}$) that are most likely given the constraints.
    * We'll let $F(\{p_i\}) = \text{# of ways to get } \{n_i\} / \text{total # of ways} = M^N$.  
    * Now do some combinatorics $\Lra$ this is a multinomial distribution (we use Stirling here: $n! \approx n\log n - n$):

        $$\begin{align}
          \log F(\{p_i\}) &= \log(N!) - \sum_{i=1}^M \log(n_i!) -     N\log M \\
          & \approx -N\log M + N\log N - \sum_{i=1}^M n_i\log(n_i) \\
          & \approx -N\log M - N \sum_{i=1}^M p_i\log(p_i)
        \end{align}$$   

        where $p_i = n_i/N$.

* This tells us that the key piece to maximize is the entropy:

    $$ S = - \sum_{i=1}^M p_i \log(p_i) . $$

* There are several arguments for maximizing the entropy:
    1. information theory, which says says maximum entropy equals minimum information (Shannon, 1948);
    2. logical consistency (Shore and Johnson, 1960);
    3. uncorrelated assignments are related monotonically to $S$ (Skilling, 1988).

* The third one tells us that unless you know specifically about correlations, it should not be in your probability assignment. One finds that entropy maximization satisfies this condition (see the notebook for a comparison of different possibilities for a test problem).

* The continuous version of entropy is

    $$
       S[p(x)] = - \int dx p(x) \log\Bigl(\frac{p(x)}{m(x)}\Bigr)
    $$    

    where $m(x)$ is a measure function. It is there to ensure that $S[p]$ is invariant under $x\rightarrow y=f(x)$. Typically this means $m(x) = $ constant.

### Example 1: Gaussian distribution
The constraints are normalization and a known variance:

$$\begin{align}
   \int_{0}^\infty dx\, p(x) &= 1, \quad x\geq 0 \\
   \int_{0}^\infty dx\, (x-\mu)^2 p(x) &= \sigma^2
\end{align}$$ 

$\Lra$ maximize

$$
  Q(p;\lambda_0,\lambda_1) = - \int dx\, p(x)\log\Bigl(\frac{p(x)}{m(x)}\Bigr) + \lambda_0 \bigl(1 - \int dx\, p(x)\bigr)
  + \lambda_1 \bigl(\sigma^2 - \int dx\, (x-\mu)^2 p(x)\bigr) ,
$$

with uniform $m(x)$ (we take $m(x) = 1$ below). The maximization is straightforward.

**Step 1:**

$$\begin{align}
  \frac{\delta Q}{\delta p(x)}
 &= -\log\frac{p(x)}{1} - \frac{p(x)}{p(x)} - \lambda_0 - \lambda_1 (x-\mu)^2 \\
  \frac{\delta Q}{\delta \lambda_0} &= 1 - \int_{-\infty}^{\infty} dx\, p(x) \quad\text{and}\quad
 \frac{\delta Q}{\delta \lambda_0} = 1 - \int_{-\infty}^{\infty} dx\, (x-\mu)^2 p(x) .
\end{align}$$

**Step 2:**

$$\begin{align}
 & \frac{\delta Q}{\delta p(x)} = 0 \Lra \log p(x) = -(1 + \lambda_0) - \lambda_1 (x-\mu)^2 \\
 & \Lra p(x) = e^{-1+\lambda_0}e^{-\lambda_1 (x-\mu)^2} .
\end{align}$$

**Step 3:**

$$\begin{align}
 & \frac{\partial Q}{\partial \lambda_0} = 0 \Lra \int_{-\infty}^{\infty} dx\, e^{-1+\lambda_0}e^{-\lambda_1 (x-\mu)^2} =
 e^{-1+\lambda_0} \sqrt{\frac{\pi}{\lambda_1}} = 1
 \Lra e^{-1+\lambda_0} = \sqrt{\frac{\lambda_1}{\pi}}   \\
 & \frac{\partial Q}{\partial \lambda_1} = 0 \Lra \int_{-\infty}^{\infty} dx\, (x-\mu)^2 e^{-1+\lambda_0}e^{-\lambda_1 (x-\mu)^2} = \underbrace{e^{-1+\lambda_0}\frac{1}{\lambda_1^{3/2}}}_{1/\sqrt{\pi}\lambda_1} \underbrace{\int_{-\infty}^{\infty} dy\, y^2 e^{-y^2}}_{\sqrt{\pi}/2} = \sigma^2
 \Lra \lambda_1 = \frac{1}{2\sigma^2}
\end{align}$$

Putting it together we get our good friend the Gaussian distribution:

$$
  p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}  e^{-(x-\mu)^2/2\sigma^2} .
$$


### Example 2: Poisson distributions

The constraints are normalization and a known mean:

$$\begin{align}
   \int_{0}^\infty dx\, p(x) &= 1, \quad x\geq 0 \\
   \int_{0}^\infty dx\, x p(x) &= \mu
\end{align}$$ 

$\Lra$ maximize

$$
  Q(p;\lambda_0,\lambda_1) = - \int dx\, p(x)\log\Bigl(\frac{p(x)}{m(x)}\Bigr) + \lambda_0 \bigl(1 - \int dx\, p(x)\bigr)
  + \lambda_1 \bigl(\mu - \int dx\, x p(x)\bigr) ,
$$

with uniform $m(x)$. The maximization is again straightforward:

$$\begin{align}
 & \frac{\delta Q}{\delta p(x)}
 = -\log\frac{p(x)}{1} - \frac{p(x)}{p(x)} - \lambda_0 - \lambda_1 x = 0 \\
 & \Lra \log p(x) = -(1 + \lambda_0) - \lambda_1 x \\
 & \Lra p(x) = e^{-1+\lambda_0}e^{-\lambda_1 x}
\end{align}$$

Finally, we determine $\lambda_0$ and $\lambda_1$ from the constraints:

$$\begin{align}
  e^{1+\lambda_0}\int_0^\infty dx\, e^{-\lambda_1 x}
    = e^{-(1+\lambda_0)}\frac{1}{\lambda_1} = 1
    \quad&\Lra\quad \lambda_1 = e^{-(1+\lambda_0)} \\
   \int_0^\infty dx\, \underbrace{e^{-(1+\lambda_0)}}_{\lambda_1}
 \underbrace{e^{-\lambda_1 x}x}_{1/\lambda_1^2}
  = \mu
  \quad&\Lra\quad \lambda_1 = \frac{1}{\mu} .
\end{align}$$

Substituting we get the Poisson distribution:
 
$$
  p(x) = \frac{1}{\mu}  e^{-x/\mu} .
$$




Try the other examples!        