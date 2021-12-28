#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid"); sns.set_context("talk")

import GPy    # a Gaussian Process (GP) framework written in python


# In[2]:


X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05


# In[3]:


np.shape(X)


# In[4]:


X = np.array([1,2,3,7,8,9])
X = np.reshape(X, (-1, 1))
Y = np.array([0,.1,.05,-.05,-.1,0])
Y = np.reshape(Y, (-1, 1))


# In[5]:


np.shape(X)


# In[6]:


#kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
kernel = GPy.kern.Matern32(1, 0.5, 0.2)


# In[7]:


#type GPy.kern.<tab> here:
#GPy.kern.BasisFuncKernel?


# In[8]:


m = GPy.models.GPRegression(X,Y,kernel)


# In[9]:


display(m)


# In[10]:


fig = m.plot()
#GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
ax=plt.gca()


# In[11]:


m.optimize(messages=True)


# In[12]:


m.optimize_restarts(num_restarts = 10)


# In[13]:


display(m)
fig = m.plot(plot_density=False, figsize=(8,6))
#GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')
#fig = m.plot(plot_density=False, figsize=(8,6))
ax=plt.gca()


# In[ ]:





# In[ ]:




