#!/usr/bin/env python
# coding: utf-8

# # Standard medical example by applying Bayesian rules of probability <span style="color: red">Key</span>
# 
# Goal: Use the Bayesian rules to solve a familiar problem.
# 
# Physicist-friendly references:
# 
# * R. Trotta, [*Bayes in the sky: Bayesian inference and model selection in cosmology*](https://www.tandfonline.com/doi/abs/10.1080/00107510802066753), Contemp. Phys. **49**, 71 (2008)  [arXiv:0803.4089](https://arxiv.org/abs/0803.4089).
#         
# * D.S. Sivia and J. Skilling, [*Data Analysis: A Bayesian Tutorial, 2nd edition*]("https://www.amazon.com/Data-Analysis-Bayesian-Devinderjit-Sivia/dp/0198568320/ref=mt_paperback?_encoding=UTF8&me=&qid="), (Oxford University Press, 2006).
#     
# * P. Gregory,
#      [*Bayesian Logical Data Analysis for the Physical Sciences: A Comparative Approach with Mathematica® Support*]("https://www.amazon.com/Bayesian-Logical-Analysis-Physical-Sciences/dp/0521150124/ref=sr_1_1?s=books&ie=UTF8&qid=1538587731&sr=1-1&keywords=gregory+bayesian"), (Cambridge University Press, 2010).
# 
# $\newcommand{\pr}{\textrm{p}} $

# ### Bayesian rules of probability as principles of logic 
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
#       p(x \mid  I) = \sum_j p(x,y_j \mid I) 
#         \quad \longrightarrow \quad
#        \color{red}{p(x \mid I) = \int\!dy\, p(x,y \mid I)} 
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
# See <a href="https://www.amazon.com/Algebra-Probable-Inference-Richard-Cox/dp/080186982X/ref=sr_1_1?s=books&ie=UTF8&qid=1538835666&sr=1-1">Cox</a> for the proof.

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
# 
#     **Seems like it should be high because the false negative rate is low.  I.e., the test doesn't often miss finding UD.**

# 2. *In the $\pr(\cdot | \cdot)$ notation, what is your ultimate goal?*
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD" 
# 
#     **You want to know if you have the disease, given that you have tested positively, therefore: $\ \ \pr(H | D)$**

# 3. *Express the false positive rate in $\pr(\cdot | \cdot)$ notation.* \[Ask yourself first: what is to the left of the bar?\]
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD"  
# 
#     **The probability that you are trying to find is that you get a positive result on the test (so $D$ should be on the left of the bar) given that you don't actually have the disease (this is the "false" part).  So $\overline{H}$ on the right. (Again, when you talk about false positive it is about the test result, not the disease, so $D$ is on the left.) Overall with the probability we are given (derived from the rate):**  $\ \ \pr(D | \overline{H}) = 0.023$
# 

# 4. *Express the false negative rate in $\pr(\cdot | \cdot)$ notation. By applying the sum rule, what do you also know? (If you get stuck answering the question, do the next part first.)* 
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD" 
# 
#     **False negative is the counterpart of false positive, so the probability of $\overline{D}$ given $H$:  $\ \ \pr(\overline{D}|H) = 0.014$.  For both false negative and false positive cases, the probability is the *outcome of the test*, given additional information. You might have been fooled by the wording above: "false negative means you have UD, but the test says you don't". This might cause you to think that $H$ should be on the left. But reword it as: "false negative means that the test says you don't have UD, but you do". This makes it clearer that the probability is about the test result, not about the disease itself.**
# 
#     **The sum rule says $\ \ \pr(D|H) + \pr(\overline{D}|H) = 1\ $, therefore we know: $\ \ \pr(D|H) = 0.986$ This probability being so close to one is what makes us think the probability we have the disease is high.**
# 

# 5. *Should $\pr(D|H) + \pr(D|\overline H) = 1$?
#     Should $\pr(D|H) + \pr(\overline D |H) = 1$?
#     (Hint: does the sum rule apply on the left or right of the $|$?)*
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD" 
# 
#     **$\pr(D|H) + \pr(D|\overline H) =  1.09 \neq 1\ \ $ so the first answer is no.  But the sum rule holds when summing over all possibilities on the *left* of the bar with the same statements on the right of the bar, which is not the case here.**
# 
#     **The second sum *does* satisfy these conditions, so we expect the sum rule to hold and $\pr(D|H) + \pr(\overline D |H) = 1$, which we've already used.**
# 

# 6. *Apply Bayes' theorem to your result for your ultimate goal (don't put in numbers yet).
#    Why is this a useful thing to do here?*
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD"  
# 
#     **Bayes' theorem with just the $p(\cdot|\cdot)$s:**
# 
#     $$
#   \pr(H|D) = \frac{\pr(D|H)\,\pr(H)}{\pr(D)}
# $$
# 
#     **This is useful because we know $\pr(D|H)$.  But we still need $\pr(H)$ and $\pr(D)$.**

# 7. Let's find the other results we need.  *What is $\pr(H)$?
#   What is $\pr(\overline H)$?*
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD"  
# 
#     **We are told that 1 in 10,000 people have the disease, so $\ \ \pr(H) = 10^{-4}$**
# 
#     **That means by the sum rule that $\ \ \pr({\overline H}) = 1 - \pr(H) = 1 - 10^{-4}$**
# 

# 8. Finally, we need $\pr(D)$.  *Apply marginalization first, and then
#   the product rule twice to get an expression for $\pr(D)$ in terms of quantities
#   we know.*
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD"
# 
#     **The strategy here is to observe that we know various probabilities with $D$ on the left of the bar and statements on the right side of the bar.  Can we combine them to get $\pr(D)$?**
# 
#     **Marginalization: $\ \ \pr(D) = \pr(D, H) + \pr(D, \overline{H})\ \ $ (recall that these are joint probabilities, not conditional probabilities).**
# 
#     **Now apply the product rule to each term: $\ \ \pr(D, H) = \pr(D|H)\, \pr(H)\ \ $ and $\ \ \pr(D,\overline{H}) = \pr(D|\overline{H})\, \pr(\overline{H})$** 
# 
#     **Put it together with numbers:**
# 
#     $$
# \pr(D) = \pr(D|H)\, \pr(H) + \pr(D|\overline{H})\, \pr(\overline{H}) = 0.986\times 10^{-4} + 0.023\times(1 - 10^{-4}) \approx 0.023
# $$
# 

# 9. *Now plug in numbers into Bayes' theorem and calculate the result.  What do you get?*
# <br>
# Notation: $H$ = "you have UD", &nbsp;&nbsp; $\overline H$ = "you do not have UD",  &nbsp;&nbsp; $D$ = "you test positive for UD", &nbsp;&nbsp;  $\overline D$ = "you test negative for UD"  
# 
#     $$\pr(H|D) = \frac{0.986 \times 0.0001}{0.023} = 0.0043$$
# 
#     **or about $0.43\%$, which is really low!**
# 
#     **We conclude this is a terrible test!  If we imagine 10000 people taking the test, the expectation is that only one of them actually has UD, but 230 will get a positive result.  We need the false positive rate to be much smaller relative to the expected rate in the population for this to be a better test. (Of course, maybe this is just an inexpensive preliminary screening and the expensive test with the low false positive rate only needs to be performed on the 230 people.)**
