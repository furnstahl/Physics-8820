#!/usr/bin/env python
# coding: utf-8

# # Assignment: 2D radioactive lighthouse location using MCMC
# 
# As before, a radioactive source that emits gamma rays randomly in time but uniformly in angle is placed at $(x_0, y_0)$.  The gamma rays are detected on the $x$-axis and these positions are saved, $x_k$, $k=1,2,\cdots, N$.  Given these observed positions, the problem is to estimate the location of the source.
# 
# Unlike before, we will not assume that $y_0$ is known. Your task is to estimate both $x_0$ and $y_0$.  Since we are using Bayesian methods, this means finding the joint posterior for $x_0$ and $y_0$, given the data on where the gamma rays were detected.
# 
# You will combine the results and Python code from the `radioactive_lighthouse_exercise.ipynb` and `parameter_estimation_Gaussian_noise.ipynb` notebooks, sampling the posterior using `emcee` and plotting it using `corner`.
# 

# ## Learning goals:
# 
# * Be able to re-use markdown and Python from existing notebooks to perform similar tasks (even if not understanding all details); e.g., generating data, sampling via MCMC, making plots.
# * Successfully apply the basic ideas of Bayesian statistics: Bayes theorem, priors, sampling of posteriors.
# * Successfully analyze results (with hints).
# * Try out markdown.
# 
# Note: you shouldn't need to recalculate anything; focus on the notebook ingredients.

# ## Expressions

# The posterior we want is:
# 
# $$ p(x_0, y_0 | \{x_k\}, I) \overset{?}{=}
# $$
# 
# *Using $\LaTeX$, fill in the right side of the equation for Bayes' rule as it applies in this case.* 
# 

# *Then add below the expression for the likelihood in this case (replace the dots with the correct form and fill in the right side):*
# 
# $$  p(\{x_k\} | \cdots) \overset{?}{=}
# $$
# 

# *Describe in words what you will use for the prior:*

# *Do you need to find an expression for the denominator pdf?  Explain.*

# ## Python imports
# 
# You will need to import emcee and corner.  The best way to do this is to follow the instructions for making an environment (see Carmen page).  An alternative is to install them separately using (at the command line):
# 
# `conda install -c astropy emcee`
# 
# `conda install -c astropy corner`

# In[1]:


# Copy to here all of the Python imports you think will be relevant


# ## Generating the data
# 
# Copy-and-paste here the code from `radioactive_lighthouse_exercise.ipynb` used to generate the $x_k$ points.  Note that you have control over where the true position is but also the size of the dataset.  You will want to adjust both later. It is recommended to add the code that plots the distribution of the data (to verify it is the same as before), but this is not required.

# In[ ]:


# Add code here to generate the dataset.  Start with 500 points.


# ## PDFs for applying Bayes' rule
# 
# *Adapt code from the two notebooks to express the logarithms of the prior, likelihood, and posterior for this case.*

# In[ ]:


# Add code for the log_prior, log_likelihood, and log_posterior


# ## Run MCMC
# 
# *Copy code to run `emcee` and make any necessary changes.*

# In[ ]:


# Add the code to run emcee


# ## Generate figures
# 
# Note: you do not need to do the maximum likelihood estimates that were added to the corner plot for the Gaussian noise case.  You can skip that part of the code.

# In[2]:


# Add the code to make a corner plot


# ## Analysis
# 
# Summarize the results for the following investigations.  You can simply add to the cell containing the questions or else start new cells.

# 1. *Run the notebook 10 times (which will generate 10 different sets of random data).  Record the values of $x_0$ and $y_0$ and their uncertainties given on the corner plot for each run.  Is the actual spread of values consistent with the given uncertainties?* <br/><br/>
# 
# 2. *Increase the number of data points by a factor of 10.  By how much does the width of the posterior shrink?  Is this expected?* <br/><br/>
# 
# 3. *What does increasing the number of MC steps per walker do?* <br/><br/>
# 
# 4. *Try changing `x0_true` and `y0_true`.  Does it still work?* <br/><br/>

# 

# 

# 

# In[ ]:




