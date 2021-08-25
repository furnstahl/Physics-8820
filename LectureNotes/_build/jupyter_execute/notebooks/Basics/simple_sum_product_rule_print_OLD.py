#!/usr/bin/env python
# coding: utf-8

# ## Checking the sum and product rules, and their consequences
# 
# Goal: Check using a very simple example that the Bayesian rules are consistent with standard probabilities based on frequencies.  Also check notation and vocabulary.
# $% Some LaTeX definitions we'll use.
# \newcommand{\pr}{\textrm{p}}
# $

# ### Bayesian rules of probability as principles of logic 
# 
# Notation: $p(x \mid I)$ is the probability (or pdf) of $x$ being true
# given information $I$
# 
# 1. **Sum rule:** If set $\{x_i\}$ is exhaustive and exclusive, 
#   $$ \sum_i p(x_i  \mid  I) = 1   \quad \longrightarrow \quad       \color{red}{\int\!dx\, p(x \mid I) = 1} 
#   $$ 
#     * cf. complete and orthonormal 
#     * implies *marginalization* (cf. inserting complete set of states or integrating out variables - but be careful!)
#   $$
#    p(x \mid  I) = \sum_j p(x,y_j \mid I) 
#     \quad \longrightarrow \quad
#    \color{red}{p(x \mid I) = \int\!dy\, p(x,y \mid I)} 
#   $$
#    
#   
# 2. **Product rule:** expanding a joint probability of $x$ and $y$         
#      $$
#          \color{red}{ p(x,y \mid I) = p(x \mid y,I)\,p(y \mid I)
#               = p(y \mid x,I)\,p(x \mid I)}
#      $$
# 
#     * If $x$ and $y$ are <em>mutually independent</em>:  $p(x \mid y,I)
#       = p(x \mid I)$, then        
#     $$
#        p(x,y \mid I) \longrightarrow p(x \mid I)\,p(y \mid I)
#     $$
#     * Rearranging the second equality yields <em> Bayes' Rule (or Theorem)</em>
#      $$
#       \color{blue}{p(x  \mid y,I) = \frac{p(y \mid x,I)\, 
#        p(x \mid I)}{p(y \mid I)}}
#      $$
# 
# 

# ### Answer the questions in *italics*. Check answers with your neighbors. Ask for help if you get stuck or are unsure.

# In[1]:


get_ipython().run_cell_magic('html', '', '<style>\n table { width:70% !important; }\n table td, th { border: 1px solid black !important; \n         text-align:center !important;\n         font-size: 16px }\n</style>\n\n\n\n\n')


# |     TABLE 1     | Blue         | Brown         |  Total        |
# | :-------------: | :----------: | :-----------: | :-----------: |
# |  Tall           | 1            | 17            | 18            |
# | Short           | 37           | 20            | 57            |
# | Total           | 38           | 37            | 75            |
# 
# |     TABLE 2     | Blue         | Brown         |  Total        |
# | :-------------: | :----------: | :-----------: | :-----------: |
# |  Tall           |      &nbsp;    |   &nbsp;        |   &nbsp;      |
# | Short           |      &nbsp;    |   &nbsp;        |   &nbsp;      |
# | Total           |      &nbsp;    |   &nbsp;        |   &nbsp;      |

# 1. Table 1 shows the number of blue- or brown-eyed and tall or short individuals in a population of 75.
# *Fill in the blanks in Table 2 with probabilities (in decimals with three places, not fractions) based on the usual "frequentist" interpretations of probability* (which would say that the probability of randomly drawing an ace from a deck of cards is 4/52 = 1/13). *Add x's in the row and/or column that illustrates the sum rule.*
# <br>
# <br>

# 2. *What is $\pr(short, blue)$? Is this a joint or conditional probability?  What is $\pr(blue)$? 
# <br>From the product rule, what is $\pr(short | blue)$?  Can you read this result directly from the table?*
# <br>
# <br>
# <br>
# <br>

# 3. *Apply Bayes' theorem to find $\pr(blue | short)$ from your answers to the last part.*
# <br>
# <br>
# <br>
# <br>

# 4. *What rule does the second row (the one starting with "Short") illustrate?  Write it out in $\pr(\cdot)$ notation.* 
# <br>
# <br>
# <br>

# 5. *Are the probabilities of being tall and having brown eyes mutually independent?  Why or why not?*
# <br>
# <br>
# <br>
# 
# 

# In[ ]:




