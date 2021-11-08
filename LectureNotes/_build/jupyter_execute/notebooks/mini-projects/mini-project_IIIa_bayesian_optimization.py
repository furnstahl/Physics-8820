#!/usr/bin/env python
# coding: utf-8

# # Mini-project IIIa: Bayesian Optimization

# ## Import modules

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid"); sns.set_context("talk")

import GPy
import GPyOpt   # This will do the Bayesian optimization


# ## 1. A univariate example with GPyOpt

# Try to minimize the function
# 
# $$
# \sin(3\theta) + \theta^2 - 0.7 \theta
# $$
# 
# on the interval $\theta \in [-1,2]$.

# a. **Plot the true function**

# In[ ]:


# Insert code here
#


# b. Find the minimum using `scipy.optimize.minimize`. Plot in the figure with the true function. Repeat with a few different seeds for the starting point. 
# 
# **Do you always get the same minimum?**
# <br><br>

# In[ ]:


# Insert code here
#


# c. Use Bayesian Optimization with GPyOpt (following the example in the lecture notebook).
# 
# **Plot the statistical model and the acquisition function for the first ten iterations. Also plot the final summary of the BayesOpt convergence sequence.**

# In[ ]:


# Insert code here
#


# d. Change the acquisition function to 'LCB'. Make sure to plot the iterations. 
# 
# **How do the acquisition functions compare when it comes to exploration-exploitation?**
# <br><br>

# In[ ]:


# Insert code here
#


# e. **Repeat with noise added to the true function when generating data.**

# * Assuming that we have an input parameter vector `X`, and that we have defined `noise = 0.2`. Then we can create some noise with normal distribution using
# ```python
# noise * np.random.randn(*X.shape)
# ```
# * Redefine your "true" function so that it returns results with such noise and repeat the `GPyOpt` implementation (see the 2-parameter example in the lecture notebook).
# * It is important that your GP expects a noisy function. You must set `exact_feval = False` in `GPyOpt.methods.BayesianOptimization`.
# * Plot several samples from the "noisy" true function (using e.g. `alpha=0.1` to make them semi-transparent). Also plot the true function without noise.
# * Perform the Bayesian optimization. Study the convergence, but also the statistical model. **How is it different compared to the statistical model in the example without noise?**
# <br><br>

# f. **Build the statistical model in BayesOpt with a different kernel.** 

# * Try in particular with the `Matern32` kernel. Do you remember what it looks like?
# * Define a GPy kernel with your initial guess variance and lengthscale
# ```python
# GPkernel = GPy.kern.Matern32(input_dim=1, variance=1.0, lengthscale=1.0)
# ```
# * Include this kernel as an input argument to `GPyOpt.methods.BayesianOptimization` 
# ```python
# optimizer = BayesianOptimization(f=fNoise, 
#                                  model_type='GP',
#                                  kernel=GPkernel, 
#                                  ...
# ```
# 

# In[ ]:


# Insert code here
#
#


# **Questions to answer:**
# * Can you decide if any of these kernels work better for this problem then the other? 
# <br><br>
# * What is the observable difference between the posterior function in this experiment compared to the previous one with the default `RBF` kernel?
# <br><br>
# * How would you decide which kernel to use for your problem?**
# <br><br>

# ## 2. Build your own BayesOpt algorithm (optional or for your project)

# Now try to repeat the above, but **assemble your own BayesOpt algorithm** using functions from `numpy`, `scipy`, and `GPy` (for building the statistical model).

# Recall the pseudo-code for BayesOpt
# 1. initial $\mathbf{\theta}^{(1)},\mathbf{\theta}^{(2)},\ldots \mathbf{\theta}^{(k)}$, where $k \geq 2$
# 1. evaluate the objective function $f(\mathbf{\theta})$ to obtain $y^{(i)}=f(\mathbf{\theta}^{(i)})$ for $i=1,\ldots,k$
# 1. initialize a data vector $\mathcal{D}_k = \left\{(\mathbf{\theta}^{(i)},y^{(i)})\right\}_{i=1}^k$
# 1. select a statistical model for $f(\mathbf{\theta})$
# 1. **for** {$n=k+1,k+2,\ldots$}
#    1.    select $\mathbf{\theta}^{(n)}$ by optimizing the acquisition function: $\mathbf{\theta}^{(n)} = \underset{\mathbf{\theta}}{\text{arg max}}\, \mathcal{A}(\mathbf{\theta}|\mathcal{D}_{n-1})$
#    1.    evaluate the objective function to obtain $y^{(n)}=f(\mathbf{\theta}^{(n)})$
#    1.    augment the data vector $\mathcal{D}_n = \left\{\mathcal{D}_{n-1} , (\mathbf{\theta}^{(n)},y^{(n)})\right\}$
#    1.    update the statistical model for $f(\mathbf{\theta})$
# 1. **end for**
# 

# **Sub-tasks:**
# * You have to implement all steps in the above pseudo-code.
# * For the statistical model you can use `GPy`, following the examples from last week's lecture and exercise. Remember that the model has to be updated at step 5D.
# * Implement the LCB acquisition function for use in step 5A. The maximum of $\mathcal{A}(\theta)$ can be found using `scipy.minimize` (note that you want the maximum...). It is a good idea to try several different starting points. See example code below, or implement your own algorithm if you prefer bug checking your own code.
#   As an alternative to LCB, if you have time, you can also try the implementation of Expected Improvement in the code snippet below. However, this code might have to be cleansed of bugs.
# 

# Example code for a function that proposes the next sampling point by computing the location of the acquisition function maximum. Optimization is restarted `n_restarts` times to avoid local optima.

# In[ ]:


from scipy.optimize import minimize

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    ''' 
    Proposes the next sampling point by optimizing the acquisition function. 
    Args: 
    acquisition: Acquisition function. 
    X_sample: Sample locations (n x d). 
    Y_sample: Sample values (n x 1). 
    gpr: A GaussianProcessRegressor fitted to samples. 
    Returns: Location of the acquisition function maximum. 
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


# Example code for the Expected Improvement acquisition function.

# In[ ]:


from scipy.stats import norm

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    ''' 
    Computes the EI at points X based on existing samples, 
    X_sample and Y_sample, using a Gaussian process surrogate model. 
    Args: 
    X: Points at which EI shall be computed (m x d). 
    X_sample: Sample locations (n x d). 
    Y_sample: Sample values (n x 1). 
    m: A GP model from GPy fitted to samples. 
    xi: Exploitation-exploration trade-off parameter. 
    
    Returns: Expected improvements at points X. 
    '''
    (mu, sigma) = gpr.predict(X)
    (mu_sample, sigma_sample) = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, X_sample.shape[1])
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


# In[ ]:


# Insert code here
#
#


# ## 3. Test on bivariate example **(Do this for a plus)**

# Use your own BayesOpt implementation, or the GPy one, to find the minimum of the following objective function:

# In[ ]:


def langermann(x):
    """
    Langermann test objective function.
    Args: 
    x: Two-dimensional point; format: [[x0, x1]] 
    Returns: Function value 
    """

    x=np.asarray(x[0]) # for compatibility with GPyOpt
    
    a = [3,5,2,1,7]
    b = [5,2,1,4,9]
    c = [1,2,5,2,3]
    
    return -sum(c*np.exp(-(1/np.pi)*((x[0]-a)**2                                + (x[1]-b)**2))*np.cos(np.pi*((x[0]-a)**2                                 + (x[1]-b)**2)))


# **Be sure to investigate different choices for the acquisition function and for the covariance function of your statistical model. In particular, be sure to compare the `RBF` and `Matern32` kernels.**

# In[1]:


# Insert code here
#
#


# ## 4. Multivariate test examples (optional)

# In case you have time, try one of the challenging multivariate test functions that are presented in the Appendix of [Bayesian optimization in ab initio nuclear physics. arXiv:1902.00941](https://arxiv.org/abs/1902.00941).

# In[ ]:


# Insert code here
#
#

