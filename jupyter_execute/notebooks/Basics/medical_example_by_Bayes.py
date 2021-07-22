#!/usr/bin/env python
# coding: utf-8

# # Standard medical example by applying Bayesian rules of probability
# 
# Goal: Use the Bayesian rules to solve a familiar problem.
# $\newcommand{\pr}{\textrm{p}} $

# ## Bayesian rules of probability as principles of logic 
# 
# Notation: $p(x \mid I)$ is the probability (or pdf) of $x$ being true
# given information $I$
# 
# 1. **Sum rule:** If set $\{x_i\}$ is exhaustive and exclusive, 
# 
#     $$ \sum_i p(x_i  \mid  I) = 1   \quad \longrightarrow \quad       \color{red}{\int\!dx\, p(x \mid I) = 1} 
#     $$ 
# 
#     * cf. complete and orthonormal 
#     * implies *marginalization* (cf. inserting complete set of states or integrating out variables - but be careful!)
# 
#     $$
#      p(x \mid  I) = \sum_j p(x,y_j \mid I) 
#        \quad \longrightarrow \quad
#       \color{red}{p(x \mid I) = \int\!dy\, p(x,y \mid I)} 
#     $$
#    
#   
# 2. **Product rule:** expanding a joint probability of $x$ and $y$         
# 
#     $$
#          \color{red}{ p(x,y \mid I) = p(x \mid y,I)\,p(y \mid I)
#               = p(y \mid x,I)\,p(x \mid I)}
#     $$
# 
#     * If $x$ and $y$ are <em>mutually independent</em>:  $p(x \mid y,I)
#       = p(x \mid I)$, then        
#     
#     $$
#        p(x,y \mid I) \longrightarrow p(x \mid I)\,p(y \mid I)
#     $$
#     
#     * Rearranging the second equality yields <em> Bayes' Rule (or Theorem)</em>
#     
#     $$
#       \color{blue}{p(x  \mid y,I) = \frac{p(y \mid x,I)\, 
#        p(x \mid I)}{p(y \mid I)}}
#     $$
# 
# See Cox for the proof.

# ## Answer the questions in *italics*. Check answers with your neighbors. Ask for help if you get stuck or are unsure.

# Suppose there is an unknown disease (call it UD) and there is a test for it.
# 
# a. The false positive rate is 2.3%. ("False positive" means the test says you have UD, but you don't.) <br>
# b. The false negative rate is 1.4%. ("False negative" means you have UD, but the test says you don't.)
# 
# Assume that 1 in 10,000 people have the disease. You are given the test and get a positive result.  Your ultimate goal is to find the probability that you actually have the disease.  We'll do it using the Bayesian rules.
# 
# We'll use the notation:
# 
# * $H$ = "you have UD"
# * $\overline H$ = "you do not have UD"  
# * $D$ = "you test positive for UD"
# * $\overline D$ = "you test negative for UD"  

# 1. *Before doing a calculation (or thinking too hard :), does your intuition tell you the probability you have the disease is high or low?*
# <br>
# <br>
# 2. *In the $\pr(\cdot | \cdot)$ notation, what is your ultimate goal?*
# <br>
# <br>
# <br>
# <br>
# 3. *Express the false positive rate in $\pr(\cdot | \cdot)$ notation.* \[Ask yourself first: what is to the left of the bar?\]
# <br>
# <br>
# <br>
# <br>
# 4. *Express the false negative rate in $\pr(\cdot | \cdot)$ notation. By applying the sum rule, what do you also know? (If you get stuck answering the question, do the next part first.)* 
# <br>
# <br>
# <br>
# <br>
# 5. *Should $\pr(D|H) + \pr(D|\overline H) = 1$?
#     Should $\pr(D|H) + \pr(\overline D |H) = 1$?
#     (Hint: does the sum rule apply on the left or right of the $|$?)*
# <br>
# <br>
# <br>
# <br>
# 6. *Apply Bayes' theorem to your result for your ultimate goal (don't put in numbers yet).
#    Why is this a useful thing to do here?*
# <br>
# <br>
# <br>
# <br>
# 7. Let's find the other results we need.  *What is $\pr(H)$?
#   What is $\pr(\overline H)$?*
# <br>
# <br>
# <br>
# <br>
# 8. Finally, we need $\pr(D)$.  *Apply marginalization first, and then
#   the product rule twice to get an expression for $\pr(D)$ in terms of quantities
#   we know.*
# <br>
# <br>
# <br>
# <br>
# 9. *Now plug in numbers into Bayes' theorem and calculate the result.  What do you get?*
# <br>
# <br>
# <br>
# <br>
