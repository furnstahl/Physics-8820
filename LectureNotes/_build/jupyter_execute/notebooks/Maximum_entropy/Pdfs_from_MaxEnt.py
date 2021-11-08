#!/usr/bin/env python
# coding: utf-8

# # MaxEnt for deriving probability distributions

# Here we derive some standard probability distribution by maximizing the entropy subject to constraints derived from information that we have.

# ## Example 1: the Gaussian

# First we take the constraint 
# \begin{equation}
# \int_{-\infty}^\infty (x-\mu)^2 p(x) dx=\sigma^2.
# \end{equation}
# 
# We also have the normalization constraint:
# \begin{equation}
# \int_{-\infty}^\infty p(x) dx=1.
# \end{equation}

# So we maximize:
# 
# \begin{equation}
# Q(p;\lambda_0,\lambda_1)=-\int p(x)\ln\left(\frac{p(x)}{m(x)}\right) \, dx + \lambda_0 \left(1-\int p(x) dx\right)+ \lambda_1\left(\sigma^2 - \int p(x) (x-\mu)^2 dx\right)
# \end{equation}
# 
# We will assume a uniform m(x).

# Step 1: differentiate with respect to p(x). What do you get?

# In[ ]:





# Step 2: set the functional derivative equal to 0. Show that the solution is:
# 
# \begin{equation}
# p(x)={\cal N} \exp(-\lambda_1 (x-\mu)^2),
# \end{equation}
# 
# where ${\cal N}=e^{-1-\lambda_0}$.

# In[ ]:





# Step 3a: Now, we impose the constraints. First, use the fact that $\int_{-\infty}^{\infty} \exp(-y^2) \, dy=\sqrt{\pi}$ to fix ${\cal N}$ (and $\lambda_0$).

# In[ ]:





# Step 3b: Second, compute $\int_{-\infty}^{\infty} y^2 \exp(-y^2) \, dy$, and use the results to show $\lambda_1 = \frac{1}{2 \sigma^2}$.

# ## Example 2: the Poisson distribution

# Now we will take a constraint on the mean (first moment):
# \begin{equation}
# \int_{0}^\infty x p(x) dx=\mu.
# \end{equation}
# 
# As usual, we also have the normalization constraint:
# \begin{equation}
# \int_{0}^\infty p(x) dx=1.
# \end{equation}

# So we maximize:
# 
# \begin{equation}
# Q(p;\lambda_0,\lambda_1)=-\int p(x)\ln\left(\frac{p(x)}{m(x)}\right) \, dx + \lambda_0 \left(1-\int p(x) dx\right)+ \lambda_1\left(\mu - \int p(x) x dx\right)
# \end{equation}
# 
# We will again assume a uniform m(x).

# Go through the steps as you did in the first example.

# In[ ]:





#  You should obtain the Poisson distribution:
# 
# \begin{equation}
# p(x)=\frac{1}{\mu} \exp\left(-\frac{x}{\mu}\right)
# \end{equation}

# ## Third example: log normal distribution

# Suppose the constraint is on the variance of $\ln x$, i.e.,
# \begin{equation}
# \int p(x)\left[\log\left(\frac{x}{x_0}\right)\right]^2 dx=\sigma^2
# \end{equation}

# Change variables to $y=\log(x/x_0)$. What is the constraint in terms of $y$?

# Now maximize the entropy, subject to this constraint, and, of course, the normalization constraint.

# In[ ]:





# You should obtain the log-normal distribution:
# 
# \begin{equation}
# p(x)=\frac{1}{\sqrt{2 \pi} x \sigma} \exp\left[-\frac{\ln^2(x/x_0)}{2 \sigma^2}\right].
# \end{equation}

# When do you think it would make sense to say that we know the variance of $\log(x)$, rather than the variance of $x$ itself?

# In[ ]:





# ## Fourth example: l1-norm

# Finally, we take the constraint on the mean absolute value of $x-\mu$: $\langle |x-\mu| \rangle=\epsilon$.

# This constraint is written as:
# 
# \begin{equation}
# \int p(x) \, |x - \mu| \, dx=\epsilon.
# \end{equation}

# Use the uniform measure, and go through the steps once again, to show that:
# \begin{equation}
# p(x)=\frac{1}{2 \epsilon} \exp\left(-\frac{|x-\mu|}{\epsilon}\right).
# \end{equation}

# In[ ]:




