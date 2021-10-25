#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# Initialize random number generator
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")


# In[3]:


# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma


# In[4]:


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y, alpha=0.6)
axes[1].scatter(X2, Y, alpha=0.6)
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X1")
axes[1].set_xlabel("X2");


# In[5]:


import pymc3 as pm

print(f"Running on PyMC3 v{pm.__version__}")


# In[6]:


basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)


# In[7]:


map_estimate = pm.find_MAP(model=basic_model)
map_estimate


# In[8]:


map_estimate = pm.find_MAP(model=basic_model, method="powell")
map_estimate


# In[9]:


with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500, return_inferencedata=False)


# In[10]:


trace["alpha"][-5:]


# In[11]:


with basic_model:
    # instantiate sampler
    step = pm.Slice()

    # draw 5000 posterior samples
    trace = pm.sample(5000, step=step, return_inferencedata=False)


# In[12]:


with basic_model:
    az.plot_trace(trace);


# In[13]:


with basic_model:
    display(az.summary(trace, round_to=2))


# In[ ]:




