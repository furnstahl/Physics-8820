#  Lecture 1

## Inference: Pass 1

How do we increase or update our knowledge? We use *inference*. The dictionary definition of inference is "the act or process of reaching a conclusion about something from known facts or evidence."
We can divide inference into two classes, deductive and inductive.

* deductive inference: cause $\Longrightarrow$ effect
* inductive inference to the best explanation: effect $\Longrightarrow$ cause

Polya, in his two-volume book on "Mathematics and Plausible Reasoning" {cite}`Polya1954-POLMAP1,Polya1954-POLMAP2`, writes: "a mathematical proof is demonstrative reasoning but the inductive evidence of the physicist, the circumstantial evidence of the lawyer, the documentary evidence of the historian and the statistical evidence of the economist all belong to plausible reasoning."
(Note: these volumes are highly recommended!)
Polya contrasts a standard deductive inference involving statements *A* and *B* (an Aristotelian syllogism): *If A implies B, and A is true, then B is true* with an inductive inference: *If A implies B, and B is true, then A is more credible*. He considers qualitative patterns of plausible inference; as physicists we seek *quantitative* inferences.

In particular, physicists need ways to:
1. quantify the strength of inductive inferences;
1. update that quantification as new data is acquired.

Bayesian statistics: Do this with pmfs or pdfs $\Longrightarrow$, which are probability mass or density functions, respectively.
* Discrete probability is pmf and continuous probability is pdf.
* We will mostly do continuous but often refer to either type as a pdf (often being sloppy and calling it a probability *distribution* function).
* To a Bayesian, (almost) everything is a pdf!

Let's use physics examples to illustrate: normalized quantum mechanical wave functions (which we square to get probabilities).
* discrete example: spin-1/2 wave function $p_{\rm up} + p_{\rm down} = 1$
* continuous example: one-dimensional particle in coordinate space
<br/>
Probability *density* at $x$: $|\psi(x)|^2 \ \Longrightarrow\ p(x)$
* Remember that this has *units* (unlike a probability)
* The probability (dimensionless) of finding $x$ in $a \leq x \leq b$ is

$$
   \text{prob}(a \leq x \leq b) = \int_a^b |\psi(x)|^2\, dx
$$

Hence $p(x) \sim [L]^{-1}$ units.
* Multidimensional normalized pdfs: wavefunction squared for particle 1 at $x_1$ and particle 2 at $x_2$:

$$
  |\Psi(x_1, x_2)|^2 \ \Longrightarrow\ p(x_1, x_2) \equiv p(\vec x) \quad
  \mbox{with}\ \vec x = \{x_1, x_2\}
$$

Alternative notation for pdfs in literature: $p(\vec x) = p(\mathbf{x}) = P(\vec x) = \text{pr}(\vec x) = \text{prob}(\vec x) = \ldots$

### Vocabulary and definitions:
* $p(x_1, x_2)$ is the *joint* probability density of $x_1$ and $x_2$

    ::::{admonition} Question
    What is the probability to find particle 1 at $x_1$ while particle 2 is *anywhere*?
    :::{admonition} Answer 
    :class: dropdown 
    $\int_{-\infty}^{+\infty} |\psi(x_1, x_2)|^2\, dx_2\ \ $ or, more generally, integrated over the domain of $x_2$.
    :::
    ::::

* General: the *marginal probability density* of $x_1$ is $p(x_1) = \int p(x_1,x_2)\,dx_2$
* *Marginalizing* = ``integrating out'' (eliminating unwanted or "nuisance" parameters)

In Bayesian statistics there are pdfs (or pmfs if discrete) for 
* fit parameters --- like slope and intercept of a line
* experimental *and* theoretical uncertainties
* hyperparameters (parameters that characterize pdfs; more on these later!)
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

Go through the Jupyter notebook [](/notebooks/Basics/Exploring_pdfs.ipynb).

:::{tip}
When you follow the [](/notebooks/Basics/Exploring_pdfs.ipynb) link, you can run the notebook on a Binder cloud server using the leftmost icon at the top-middle-right, or you can download the notebook to run locally using the rightmost icon at the top-middle-right. Ultimately you should clone the github repository by following the github icon <img src="/_images/GitHub-Mark-32px.png" alt="github download icon" width="20px"> to be able to run locally and update with a simple `git pull` command.

When running on Binder, be patient; it may take a while to generate the page if the environment needs to be created from scratch (in general it is cached, so it will be much faster if others have been recently using notebooks from the repository).
:::

Points of interest:
* Importing packages: `scipy.stats`, `numpy`, `matplotlib`. Convenient abbreviations (like `np`) are introduced.
* corner is not included in Anaconda $\Longrightarrow$ use the package manager `conda` to install locally (unless you are on Binder or have used the conda environment; more later). 
    :::{tip}
    Google "conda corner" to find the command needed $\Longrightarrow$ look for `Corner::Anaconda Cloud` $\Longrightarrow$ `conda install -c astropy corner`
    :::
* scipy.stats $\Longrightarrow$ look at manual page by googling
* Come back and look at definitions
* Look at the examples: *not everything is a Gaussian distribution!*
    * You will look at the Student t pdf on your own
    :::{note}
    Trivia: "Student" was the pen name of the Head Brewer at Guiness --- a pioneer of small-sample experimental design (hence not necessarily Gaussian). His real name was William Sealy Gossett. 
    :::
* Look at projected posterior plots, which use the corner package.
    * What do you learn from the plots? (e.g., are the quantities *correlated*? How can you tell? More later!)
    * Note that these are *samples* from the pdf. We will have much to say about sampling.
* One-dimensional pdfs: note the fluctuations, larger for smaller numbers of samples.

Many follow-ups are possible, but let's first put some other Bayesian notation on the table.


## Manipulating pdfs: Bayesian rules of probability as principles of logic

* You will show these rules are consistent with standard probabilities based on frequencies in [simple_sum_product_rule.ipynb](/notebooks/Basics/simple_sum_product_rule.ipynb).

* Notation: 

$$ 
   p(A|B) \equiv \text{"probability of $A$ given $B$ is true''}
$$

* $\Longrightarrow$ *conditional* probability
* We also can say "probability of $A$ *contingent* on $B$"
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
* exhaustive, exclusive $\Longrightarrow$ cf. complete, orthonormal (includes all values and there is no overlap between members of the set).
* the sum rule *implies* marginalization 

    \begin{align}
      p(x|I) = \sum_j p(x, y_j | I) \qquad\Longrightarrow\qquad p(x|I) = \int dy\, p(x,y | I)
    \end{align}

    * We will use marginalization a lot!
    :::{warning}
    Physicists might compare marginalization to inserting complete set of states or integrating out variables. It's ok to use this as a mnemonic but be careful, this analogy breaks down in general.
    :::

:::{note}
A rule from probability says $p(A \cup B) = p(A) + p(B) - p(A \cap B)$. (That is, to calculate the probability of the union of $A$ and $B$ we need to subtract the probability of the intersection from the sum of probabilities.) This may seem to contradict our marginalization rule. The difference is that if $A$ and $B$ are *exclusive*, as we assume, then $p(A \cap B) = 0$.
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

### Your task: complete [simple_sum_product_rule.ipynb](/notebooks/Basics/simple_sum_product_rule.ipynb)

* Fill in the table based on knowledge/intuition of probabilities as frequencies (these are *estimates* of population probabilities).
* Apply the sum and product rules as directed.
* Work together and check answers with each other.

**Ask questions!**

