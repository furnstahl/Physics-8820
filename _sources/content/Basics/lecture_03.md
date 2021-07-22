#  Lecture 3

## Step through the medical example key

Some of the answers are straightforward but make sure you agree. We'll split out a few follow-up points.

::::{admonition} Follow-up question on 2.
Why is it $p(H|D)$ and not $p(H,D)$?
:::{admonition} Answer
:class: dropdown 
Recall that $p(H,D) = p(H|D) \cdot p(D)$. You are generally interested in $p(H|D)$.
If you know $p(D) = 1$, then they are the same.
:::
::::

::::{admonition} Follow-up question on 5.
The emphasis here is on the sum rule. Why didn't a column in the notebook add to 1?
:::{admonition} Answer
:class: dropdown 
Because were were looking at $p(\text{tall,blue}) + p(\text{short,blue}) \neq 1$, whereas $p(\text{tall}| \text{blue}) + p(\text{short}| \text{blue}) = 1$.
:::
::::

In general we emphasise the usefulness of Bayes' theorem to express $p(H|D)$ in terms of $p(D|H)$. Make sure that 8. and 9. are clear to you. In 8., this is standard but not so obvious at first; after it becomes familiar you will find that you jump right to the end.

## Recap of coin flipping notebook

Recall the names of the pdfs in Bayes' theoem: posterior, likelihood, prior, evidence; and recall Bayesian updating: prior + data $\rightarrow$ posterior $\rightarrow$ new prior.

Take-aways and follow-up questions from coin flipping:
1. Different priors *eventually* give the same posterior with enough data. This is called *Bayesian convergence*. How many tosses are enough? Hit `New Data` multiple times to see the fluctuations. Clearly it depends on $p_h$ and how close you want the posteriors to be. How about for $p_h = 0.4$ and $0.9$?
    :::{admonition} Answer
    :class: dropdown
    * $p_h = 0.4$ $\Longrightarrow$ $\approx 200$ tosses will get you most of the way.
    * $p_h = 0.9$ $\Longrightarrow$ much longer for the informative prior than the others.
    :::
1. Why does the "anti-prior" work well even though its dominant assumptions (most likely $p_h = 0$ or $1$) are proven wrong early on?     
    :::{admonition} Answer
    :class: dropdown
    "Heavy tails" mean it is like uniform (renormalized!) after the ends are eliminated. An important less for formulating priors: allow for deviations from your expectations.
    :::
1. Case I and Case II. From the code: `y_i = stats.beta.pdf(x,alpha_i + heads, beta_i + N - heads)` [move earlier?]
1. Is there a difference between updating sequentially or all at once? Do the simplest problem first: two tosses.
Let results be $D = \{D_k\}$ (in practice take 0's and 1's as the two choices $\Longrightarrow$ $R = \sum_k D_k$).
    * The general relation is $p(p_h | \{D_k\},I) \propto p(\{D_k\}|p_h,I) p(p_h|I)$ by Bayes' theorem.
    * First $k=1$: 
        
        $$ p(p_h | D_1,I) \propto p(D_1|p_h,I) p(p_h|I)$$ (eq:k_eq_1)
    
    * Now $k=2$:
    
        $$\begin{align}
        p(p_h|D_2, D_1) &\propto p(D_2, D_1|p_h, I)p(p_h|I) \\
             &\propto p(D_2|p_h,D_1,I) p(p_h|D_1,I) \\
             &\propto p(D_2|p_h,I)p(p_h|D_1,I) \\
             &\propto p(D_2|p_h,I)p(D_1|p_h,I)p(p_h,I)
        \end{align}$$ (eq:k_eq_2)
    
        :::{admonition} What is the justification for each step?
        :class: dropdown 
        * 1st line: Bayes' Rule
        * 2nd line: Bayes' Rule (think of $D_1 \in I'$!)
        * 3rd line: tosses are independent
        * 4th line: Bayes' Rule on the last term in the 3rd line
        :::
        The third line of {eq}`eq:k_eq_2` is the sequential result! (The prior for the 2nd flip is the posterior {eq}`eq:k_eq_1` from the first flip.)
    * So all at once is the same as sequential as a function of $p_h$, when normalized.
    * To go to $k=3$:

        $$\begin{align}
        p(p_h|D_3,D_2,D_1,I) &\propto p(D_3|p_h,I) p(p_h|D_2,D_1,I) \\
           &\propto p(D_3|p_h,I) p(D_2|p_h,I) p(D_1|p_h,I) p(p_h)
        \end{align}$$

        and so on.

1. What about "bootstrapping"?



