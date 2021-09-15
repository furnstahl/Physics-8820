#!/usr/bin/env python
# coding: utf-8

# # Assignment: Follow-ups to Parameter Estimation notebooks 
# 
# **Goal:** work through the `parameter_estimation_fitting_straight_line_I.ipynb` and `amplitude_in_presence_of_background.ipynb` notebooks, doing some of the suggested tasks and answering selected questions, as detailed below.  

# ### Learning goals:
# * Improve your understanding of the Python code in our sample notebooks.
# * Be able to articulate basic scaling of fluctuations, the signature of correlations, and the role of priors.
# * Gain experience with a basic Bayesian experimental design problem.
# * Explore an example of the central limit theorem both analytically and numerically.

# ## A. Parameter estimation example: fitting a straight line I
# 
# 1. Step through the notebook and make sure you understand the notation and calculations.  Try writing the requested pieces of Python code; if you have difficulty, study the supplied code instead and identify what parts you don't understand.  (You don't need to hand in anything for this part but please list here any questions you have.)
# <br><br><br><br>
# 
# 1. Do exercise 3: Change the random number seed to get different results and comment on how the maximum likelihood results fluctuate. How is the typical size of fluctuations related to the number of data points $N$ and the data error standard deviation $dy$? E.g., are they proportional to $N$ or $dy^2$ or what?  (Try changing $N$ and $dy$ to test your answer!)
# <br><br><br><br>
# 
# 1. In both sets of joint posterior graphs, are the slope and intercept correlated?  How do you know? Explain how they get correlated.
# <br><br><br><br>
# 
# 1. For the first set of data, answer the question: "What do you conclude about how the form of the prior affects the final posterior in this case?"
# <br><br><br><br>
# 
# 1. For the second set of data, answer the question: "Why in this case does the form of the prior have a clear effect?"  You should consider both the size of the error bars and the number of data points (and try changing them to verify the impact).
# <br><br><br><br>
# 

# ## B. Amplitude of a signal in the presence of background
# 
# 1. Step through the notebook and make sure you understand the problem and its analysis. You may find the discussion in Section 3.1 of *Sivia and Skilling, Data Analysis: A Bayesian Tutorial* useful (see Carmen modules). You don't need to hand in anything for this part but please list here any questions you have.
# <br><br><br><br>
# 
# 1. Do the "Follow-ups": 
#    * *Try both smaller and larger values of D and note the transition in the form of the pdf.*
#    * At $D=12.5$ the pdf is already looking like a Gaussian (or what most of us imagine a Gaussian to look like).  *Prove that in the limit $D \rightarrow \infty$ that* 
#    
# $$
#  p(N \mid D) \stackrel{D\rightarrow\infty}{\longrightarrow} \frac{1}{\sqrt{2\pi D}}e^{-(N-D)^2/(2D)}
# $$
# 
# You'll want to use Stirling's formula:  $x! \rightarrow \sqrt{2\pi x}e^{-x} x^x$ as $x\rightarrow\infty$.
# \[Hint: let $x = N = D(1+\delta)$ where $D\gg 1$ and $\delta \ll 1$.  And use $(1+\delta)^a = e^{a\ln (1+\delta)}$.\]
# <br><br><br><br>
#    * *Show that this limit works in practice and visualize how close it is by adding the Gaussian pdf to the plot.* (See [scipy.stats.norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) or define a function yourself.)  **Add code below to make a figure here with the limiting Poisson histogram and the limiting Gaussian pdf.**
# <br><br><br><br>
# 
# 1. Based on your observations in running the different cases (and any additional ones you add), how should you optimally design an experiment to detect the amplitude of the signal given limited resources?  For example: How many counts are needed? How should you bin the data? What $(x_k)_{\rm max}$ should you use? 
# <br><br><br><br>
# 

# In[ ]:




