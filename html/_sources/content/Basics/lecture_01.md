#  Lecture 1

## Inference: Pass 1

* deductive inference: cause $\Longrightarrow$ effect
* inference to best explanation: effect $\Longrightarrow$ cause

Scientists need ways to:
1. quantify the strength of inductive inferences
1. update that quantification as new data is acquired

Bayesian: Do this with pmfs or pdfs $\Longrightarrow$ probability mass or density functions
* Discrete is pmf and continuous is pdf
* We will mostly do continuous and call everything a pdf (often being sloppy and calling it a probability distribution function).
* To a Bayesian, everything is a pdf!

Let's use physics examples to illustrate: normalized wave functions squared.
* discrete example: spin-1/2 wave function $p_{\rm up} + p_{\rm down} = 1$
* continuous example: one-dimensional particle in coordinate space

Probability *density* at $x$: $|\psi(x)|^2 \ \Longrightarrow\ p(x)$
* Remember that this has *units* (unlike a probability)
* The probability (dimensionless) of finding $x$ in $a \leq x \leq b$ is

$$
   \text{prob}(a \leq x \leq b) = \int_a^b |\psi(x)|^2\, dx
$$

Here $p(x) \sim [L]^{-1}$ units.
* Multidimensional normalized pdfs: wavefunction squared for particle 1 at $x_1$ and particle 2 at $x_2$:

$$
  |\Psi(x_1, x_2)|^2 \ \Longrightarrow\ p(x_1, x_2) \equiv p(\vec x) \quad
  \mbox{with}\ \vec x = \{x_1, x_2\}
$$

Alternative notation for pdfs in literature: $p(\vec x) = P(\vec x) = \text{pr}(\vec x) = \text{prob}(\vec x) = \ldots$

Vocabulary and definitions:
* $p(x_1, x_2)$ is the *joint* probability density of $x_1$ and $x_2$

    ::::{admonition} Question
    What is the probability to find particle 1 at $x_1$ while particle 2 is *anywhere*?
    :::{admonition} Answer 
    :class: dropdown 
    $\int_{-\infty}^{+\infty} |\psi(x_1, x_2)|^2\, dx_2$ or, more generally, integrated over     the domain of $x_2$.
    :::
    ::::

* General: the *marginal probability density* of $x_1$ is $p(x_1) = \int p(x_1,x_2)\,dx_2$
* *Marginalizing* = ``integrating out'' (eliminating unwanted or "nuisance" parameters)

In Bayesian statistics there are pdfs (or pmfs if discrete) for 
* fit parameters --- like slope and intercept of a line
* experimental *and* theoretical uncertainties
* hyperparameters (more on these later!)
* events ("Will it rain tomorrow?")
* and much more

    ::::{admonition} Question
    What is the pdf $p(x)$ if we know **definitely** that $x = x_0$ (i.e., fixed)?
    :::{admonition} Answer 
    :class: dropdown 
    $p(x) = \delta(x-x_0)\quad$  [Note that $p(x)$ is normalized.]
    :::
    ::::


## Visualizing pdfs

Go through Exploring_pdfs.ipynb.

Points of interest:
* Importing packages: `scipy.stats`, `numpy`, `matplotlib`. Convenient abbreviations (like `np`) are introduced.
* corner is not included in Anaconda $\Longrightarrow$ use the package manager `conda` to install. 
    :::{tip}
    Google "conda corner" to find the command needed $\Longrightarrow$ look for `Corner::Anaconda Cloud` $\Longrightarrow$ `conda install -c astropy corner`
    :::
* scipy.stats $\Longrightarrow$ look at manual page
* Come back and look at definitions
* Look at the examples: *not everything is a Gaussian distribution!*
    * You will look at the Student t pdf on your own
    :::{note}
    Trivia: "Student" was the pen name of the Head Brewer at Guiness --- a pioneer of small-sample experimental design (hence not necessarily Gaussian). His real name was William Sealy Gossett. 
    :::
* Look at projected posterior plots using the corner package.
    * What do you learn from the plots?
    * Note that these are *samples* from the pdf. We will have much to say about sampling.
* One-dimensional pdfs: note the fluctuations, larger for smaller numbers of samples.

Many follow-ups are possible, but let's first put some other Bayesian notation on the table.


## Manipulating pdfs: Bayesian rules of probability as principles of logic

* You will show these rules are consistent with standard probabilities based on frequencies in simple_sum_product_rule.ipynb.

* Notation: 

$$ 
   p(A|B) \equiv \text{"probability of $A$ given $B$ is true''}
$$

* $\Longrightarrow$ *conditional* probability
* For a Bayesian, $A$ and $B$ could stand for almost anything.
* Examples: 
    * $p(\text{"below zero temperature''} | \text{"it is January in Ohio''} )$
    * $p(x | \mu, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x-\mu)^2/\sigma^2}\quad$ [Gaussian or normal distribution]
:::{note}
   $p(A | B) \neq p(A,B)\quad$ [conditional probability $\neq$ joint probability]
:::

In the following examples, $p(x|I)$ is the probability or pdf of $x$ being true given information $I$ (which could be almost anything).

### **Sum rule:**
If the set $\{x_i\}$ is *exhaustive* and *exclusive*

\begin{align}
  & \sum_i p(x_i | I ) = 1 \qquad \Longrightarrow \qquad & \int dx\, p(x|I) = 1 \\
  & \text{(discrete like spins)}       & \text{(continuous)}
\end{align}

* i.e., the sum of probabilities is equal to one.
* exhaustive, exclusive $\Longrightarrow$ cf. complete, orthonormal
* *implies* marginalization 

    \begin{align}
      p(x|I) = \sum_j p(x, y_j | I) \qquad\Longrightarrow\qquad p(x|I) = \int dy\, p(x,y | I)
    \end{align}

    * We will use marginalization a lot!
    :::{warning}
    You might compare marginalization to inserting complete set of states or integrating out variables. It's ok to use this as a mnemonic but be careful, this analogy breaks down in general.
    :::

:::{note}
A rule from probability says $p(A \cup B) = p(A) + p(B) - p(A \cap B)$. (That is, to calculate the union of $A$ and $B$ we need to subtract the intersection from the sum.) This may seem to contradict our marginalization rule. The difference is that if $A$ and $B$ are *exclusive*, as we assume, then $p(A \cap B) = 0$.
:::

### **Product rule:**

Expanding a joint probability of $x$ and $y$

$$  
   p(x,y | I) = p(x | y, I) p(y,I) = p(y| x,I) p(x,I)
$$ (eq:joint_prob)

* Note the symmetry between the first and second equalities.
* If $x$ and $y$ are *mutually independent*, then $p(x | y,I) = p(x | I)$ and

    \begin{equation}
         p(x,y | I) = p(x|I) \times p(y | I)
    \end{equation}

* Rearranging the 2nd equality in {eq}`eq:joint_prob` yields **Bayes' Rule** (or Theorem):

    \begin{equation}
         p(x | y,I) = \frac{p(y|x,I) p(x|I)}{p(y | I)}
    \end{equation}

    * Tells us how to reverse the conditional: $p(x|y) \rightarrow p(y|x)$
    * Tells us how to *update* expectations (more to come!)

A proof that the Sum and Product rules follow in any consistent implementation of probabilistic reasoning is given by Cox {cite}`Cox:1961`.

### Your task: complete simple_sum_product_rule.ipynb

* Fill in the table based on knowledge/intuition of probabilities as frequencies (these are *estimates* of population probabilities).
* Apply the sum and product rules as directed.
* Work together and check answers with each other.

**Ask questions!**

