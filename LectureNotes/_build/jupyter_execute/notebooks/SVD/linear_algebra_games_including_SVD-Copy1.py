#!/usr/bin/env python
# coding: utf-8

# # Linear algebra games including SVD for PCA
# 
# Some parts adapted from [Computational-statistics-with-Python.ipynb](https://github.com/cliburn/Computational-statistics-with-Python), which is itself from a course taught at Duke University; other parts from Peter Mills' [blog](https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254).  
# 
# The goal here is to practice some linear algebra manipulations by hand and with Python, and to gain some experience and intuition with the Singular Value Decomposition (SVD).
# $\newcommand{\Amat}{\mathbf{A}} \newcommand{\AmatT}{\mathbf{A^\top}}
# \newcommand{\thetavec}{\boldsymbol{\theta}}
# \newcommand{\Sigmamat}{\mathbf{\Sigma}}
# \newcommand{\Yvec}{\mathbf{Y}}
# $

# ## Preliminary exercise: manipulations using the index form of matrices
# 
# Warm up (you may already have done this earlier in the course): prove that the Maximum Likelihood Estimate (MLE) for $\chi^2$ given by 
# 
# $$
# \chi^2 = (\Yvec - \Amat\thetavec)^{\mathbf{\top}} \Sigmamat^{-1} (\Yvec - \Amat\thetavec)
# $$
# 
# is 
# 
# $$
# \thetavec_{\mathrm{MLE}} = (\AmatT \Sigmamat^{-1} \Amat)^{-1} (\AmatT \Sigmamat^{-1} \Yvec)  \;.
# $$
# 
# Here $\thetavec$ is a $m\times 1$ matrix of parameters (i.e., there are $m$ parameters), $\Sigmamat$ is the $m\times m$ covariance matrix, $\Yvec$ is a $N\times 1$ matrix of observations (data), and $\Amat$ is an $N\times m$ matrix 
# 
# $$
# \Amat = 
# \left(
# \begin{array}{cccc}
#    1  & x_1  & x_1^2 & \cdots \\
#    1  & x_2  & x_2^2 & \cdots \\
#    \vdots & \vdots & \vdots &\cdots \\
#    1  & x_N  & x_N^2 & \cdots
# \end{array}
# \right)
# $$
# 
# where $N$ is the number of observations.  The idea is to do this with explicit indices for vectors and matrices, using the Einstein summation convention.  
# 
# A suggested approach:
# * Write $\chi^2$ in indices: $\chi^2 = (Y_i - A_{ij}\theta_j)\Sigma^{-1}_{ii'}(Y_{i'}- A_{i'j'}\theta_{j'})$, where summations over repeated indices are implied (be careful of transposes and using enough independent indices).  *How do we see that $\chi^2$ is a scalar?*
# * Find $\partial\chi^2/\partial \theta_k = 0$ for all $k$, using $\partial\theta_j/\partial\theta_k = \delta_{jk}$. Isolate the terms with one component of $\thetavec$ from those with none.
# * You should get the matrix equation $ (\AmatT \Sigmamat^{-1} \Yvec) = (\AmatT \Sigmamat^{-1} \Amat)\thetavec$. At this point you can directly solve for $\thetavec$. *Why can you do this now?*
# * If you get stuck, see the [Lecture 13 notes](https://furnstahl.github.io/Physics-8820/content/Why_Bayes_is_better/lecture_13.html).

# ## SVD basics
# 
# Note: we will used the "reduced SVD" here, for which the matrix of singular values is square. The formal discussion of the SVD usually works with the "full SVD", the "reduced SVD" is more typically used in practice. 
# 
# A singular value decomposition (SVD) decomposes a matrix $A$ into three other matrices (we'll skip the boldface font here):
# 
# $$
# A = U S V^\top
# $$
# 
# where (take $m > n$ for now)
# * $A$ is an $m\times n$ matrix;
# * $U$ is an $m\times n$ (semi)orthogonal matrix;
# * $S$ is an $n\times n$ diagonal matrix;
# * $V$ is an $n\times n$ orthogonal matrix.
# 
# Comments and tasks:
# * *Verify that these dimensions are compatible with the decomposition of $A$.*  
# * The `scipy.linalg` function `svd` has a Boolean argument `full_matrices`.  If `False`, it returns the decomposition above with matrix dimensions as stated, which is the "reduced SVD".  If `True`, then $U$ is $m\times m$, $S$ is $m \times n$, and $V$ is $n\times n$.  We will use the `full_matrices = False` form here.  *Can you see why this is ok?*
# * Note that semi-orthogonal means that $U^\top U = I_{n\times n}$ and orthogonal means $V V^\top = V^\top V = I_{n\times n}$.  
# * In index form, the decomposition of $A$ is $A_{ij} = U_{ik} S_k V_{jk}$, where the diagonal matrix elements of $S$ are 
# $S_k$ (*make sure you agree*).
# * These diagonal elements of $S$, namely the $S_k$, are known as **singular values**.  They are ordinarily arranged from largest to smallest.
# * $A A^\top = U S^2 U^\top$, which implies (a) $A A^\top U = U S^2$.
# * $A^\top A = V S^2 V^\top$, which implies (b) $A^\top A V = V S^2$.
# * If $m > n$, we can diagonalize $A^\top A$ to find $S^2$ and $V$ and then find $U = A V S^{-1}$.  If $m < n$ we switch the roles of $U$ and $V$.
# 
# Quick demonstations for you to do or questions to answer:
# * *Show from equations (a) and (b) that $U$ is semi-orthogonal and $V$ is orthogonal and that the eigenvalues, $\{S_i^2\}$, are all positive.*
# * *Show that if $m < n$ there will be at most $m$ non-zero singular values.* 
# * *Show that the eigenvalues from equations (a) and (b) must be the same.*
# 
# A key feature of the SVD for us here is that the sum of the squares of the singular values equals the total variance in $A$, i.e., the sum of squares of all matrix elements (squared Frobenius norm). Thus the size of each says how much of the total variance is accounted for by each singular vector.  We can create a truncated SVD containing a percentage (e.g., 99%) of the variance:
# 
# $$
#   A_{ij} \approx \sum_{k=1}^{p} U_{ik} S_k V_{jk}
# $$
# 
# where $p < n$ is the number of singular values included. Typically this is not a large number.

# ### Solving matrix equations with SVD
# 
# We can solve for $\mathbf{x}$:
# 
# $$\begin{align}
#   A \mathbf{x} &= b \\
#   \mathbf{x} &= V S^{-1} U^\top b
# \end{align}$$
# 
# or $x_i = \sum_j \frac{V_{ij}}{S_j} \sum_k U_{kj} b_k$.  The value of this solution method is when we have an ill-conditioned matrix, meaning that the smallest eigenvalues are zero or close to zero.  We can throw away the corresponding components and all is well! See [also](https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/MathsMethodsNotes/L3_linear_algebra3.pdf). 
# 
# Comments:
# - If we have a non-square matrix, it still works. If $m\times n$ with $m > n$, then only $n$ singular values.
# - If $m < n$, then only $m$ singular values.
# - This is like solving 
# 
# $$A^\top A \mathbf{x} = A^\top b$$
# 
# which is called the *normal equation*.  It produces the solution to $\mathbf{x}$ that is closest to the origin, or
# 
# $$
#   \min_{\mathbf{x}} |A\mathbf x - b| \;.
# $$
# 
# **Task:** *prove these results (work backwards from the last equation as a least-squares minimization)*.

# ### Data reduction
# 
# For machine learning (ML), there might be several hundred variables but the algorithms are made for a few dozen.  We can use SVD in ML for variable reduction.  This is also the connection to sloppy physics models.  In general, our matrix $A$ can be closely approximated by only keeping the largest of the singular values.  We'll see that visually below using images.

# ## Python imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from sklearn.decomposition import PCA


# *Generate random matrices and verify the properties for SVD given above.  Check what happens when $m > n$.*

# In[8]:


A = np.random.rand(9, 9)
print('A = ', A)

Ap = np.random.randn(5, 3)
print('Ap = ', Ap)


# Check the definition of `scipy.linalg.svd` with shift-tab-tab.

# In[9]:


# SVD from scipy.linalg
U, S, V_trans = la.svd(A, full_matrices=False)
Up, Sp, Vp_trans = la.svd(Ap, full_matrices=False)


# In[10]:


print(U.shape, S.shape, V_trans.shape)


# In[11]:


# Transpose with T, matrix multiplication with @
print(U.T @ U)


# In[12]:


# Here's one way to suppress small numbers from round-off error
np.around(U.T @ U, decimals=15)


# In[13]:


# Predict this one before evaluating!
print(U @ U.T)


# Go on and check the other claimed properties.  
# 
# For example, is $A = U S V^\top$? (Note: you'll need to make $S$ a matrix with `np.diag(S)`.)

# In[19]:


print(A)


# In[16]:


# Check the other properties, changing the matrix size and shapes.
print(A - U@np.diag(S)@V_trans)


# For a square matrix, compare the singular values in $S$ to the eigenvalues from `la.eig`.  What do you conclude?  Now try this for a symmetric matrix (note that a matrix plus its transpose is symmetric).

# In[22]:


Aval, Avec = la.eig(A)


# In[23]:


print(S, Aval)


# ## SVD applied to images for compression
# 
# Read in `../../_images/elephant.jpg` as a gray-scale image. The image has $1066 \times 1600$ values. Using SVD, recreate the image with a relative error of less than 0.5%. What is the relative size of the compressed image as a percentage?

# In[24]:


from skimage import io

img = io.imread('../../_images/elephant.jpg', as_gray=True)
plt.imshow(img, cmap='gray');
print('shape of img: ', img.shape)


# In[25]:


# turn off axis
plt.imshow(img, cmap='gray')
plt.gca().set_axis_off()


# In[26]:


# Do the svg
U, S, Vt = la.svd(img, full_matrices=False)


# In[27]:


# Check the shapes
U.shape, S.shape, Vt.shape


# In[28]:


# Check that we can recreate the image
img_orig = U @ np.diag(S) @ Vt
print(img_orig.shape)
plt.imshow(img_orig, cmap='gray')
plt.gca().set_axis_off()


# Here's how we can efficiently reduce the size of the matrices.  Our SVD should be sorted, so we are keeping only the largest singular values up to a point.

# In[29]:


# Pythonic way to figure out when we've accumulated 99.5% of the result
k = np.sum(np.cumsum((S**2)/(S**2).sum()) <= 0.995)


# #### Aside: dissection of the Python statement to find the index for accumulation

# In[30]:


test = np.array([5, 4, 3, 2, 1])
threshold = 0.995
print('initial matrix, in descending magnitude: ', test)
print( 'fraction of total sum of squares: ', (test**2) / (test**2).sum() )
print( 'cumulative fraction: ', np.cumsum((test**2) / (test**2).sum()) )
print( 'mark entries as true if less than or equal to threshold: ',
       (np.cumsum((test**2) / (test**2).sum()) <= threshold) )
print( 'sum up the Trues: ',
       np.sum(np.cumsum((test**2) / (test**2).sum()) <= threshold) )
print( 'The last result is the index we are looking for.')


# In[31]:


# Let's plot the eigenvalues and mark where k is
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.semilogy(S, color='blue', label='eigenvalues')
ax.axvline(k, color='red', label='99.5% of the variance');
ax.set_xlabel('eigenvalue number')
ax.legend()
fig.tight_layout()


# Now keep only the most significant eigenvalues (those up to k).

# In[34]:


img2 = U[:,:k] @ np.diag(S[:k])@ Vt[:k, :]
img2.shape


# In[35]:


plt.imshow(img2, cmap='gray')
plt.gca().set_axis_off();


# In[36]:


k99 = np.sum(np.cumsum((S**2)/(S**2).sum()) <= 0.99)
img99 = U[:,:k99] @ np.diag(S[:k99])@ Vt[:k99, :]


# In[37]:


plt.imshow(img99, cmap='gray')
plt.gca().set_axis_off();


# Let's try another interesting picture . . .

# In[38]:


fraction_kept = 0.995

def svd_shapes(U, S, V, k=None):
    if k is None:
        k = len(S)
    U_shape = U[:,:k].shape
    S_shape = S[:k].shape
    V_shape = V[:,:k].shape
    print(f'U shape: {U_shape}, S shape: {S_shape}, V shape: {V_shape}')


img_orig = io.imread('../../_images/Dick_in_tailcoat.jpg')
img = io.imread('../../_images/Dick_in_tailcoat.jpg', as_gray=True)

U, S, V = la.svd(img)
svd_shapes(U, S, V)

k995 = np.sum(np.cumsum((S**2)/(S**2).sum()) <= fraction_kept)
print(f'k995 = {k995}')
img995 = U[:,:k995] @ np.diag(S[:k995])@ V[:k995, :]
print(f'img995 shape = {img995.shape}')
svd_shapes(U, S, V, k995)

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(img_orig)
ax1.set_axis_off()

ax2 = fig.add_subplot(1,3,2)
ax2.imshow(img, cmap='gray')
ax2.set_axis_off()

ax3 = fig.add_subplot(1,3,3)
ax3.imshow(img995, cmap='gray')
ax3.set_axis_off()

fig.tight_layout()


# In[39]:


# Let's plot the eigenvalues and mark where k is
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.semilogy(S, color='blue', label='eigenvalues')
ax.axvline(k995, color='red', label='99.5% of the variance');
ax.set_xlabel('eigenvalue number')
ax.legend()
fig.tight_layout()


# ### Things to do:
# 
# * Get your own figure and duplicate these results.  Then play!
# * As you reduce the percentage of the variance kept, what features of the image are retained and what are lost?
# * See how small you can make the percentage and still recognize the picture.
# * How is this related to doing a spatial Fourier transform, applying a low-pass filter, and transforming back.  (Experts: try this!)

# In[ ]:





# ## Covariance, PCA and SVD

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

np.set_printoptions(precision=3)


# Recall the formula for covariance
# 
# $$
# \text{Cov}(X, Y) = \frac{\sum_{i=1}^n(X_i - \bar{X})(Y_i - \bar{Y})}{n-1}
# $$
# 
# where $\text{Cov}(X, X)$ is the sample variance of $X$.

# In[41]:


def cov(x, y):
    """Returns covariance of vectors x and y)."""
    xbar = x.mean()
    ybar = y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)


# In[42]:


X = np.random.random(10)
Y = np.random.random(10)


# In[43]:


np.array([[cov(X, X), cov(X, Y)], [cov(Y, X), cov(Y,Y)]])


# In[44]:


np.cov(X, Y)  # check against numpy


# In[45]:


# Extension to more variables is done in a pair-wise way
Z = np.random.random(10)
np.cov([X, Y, Z])


# ### Eigendecomposition of the covariance matrix

# In[46]:


# Zero mean but off-diagonal correlation matrix
mu = [0,0]
sigma = [[0.6,0.2],[0.2,0.2]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, n).T
plt.scatter(x[0,:], x[1,:], alpha=0.2);


# In[47]:


# Find the covariance matrix of the matrix of points x
A = np.cov(x)


# In[48]:


# m = np.array([[1,2,3],[6,5,4]])
# ms = m - m.mean(1).reshape(2,1)
# np.dot(ms, ms.T)/2


# In[49]:


# Find the eigenvalues and eigenvectors
e, v = la.eigh(A)


# In[50]:


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(x[0,:], x[1,:], alpha=0.2)
for e_, v_ in zip(e, v.T):
    ax.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
ax.axis([-3,3,-3,3])
ax.set_aspect(1)
ax.set_title('Eigenvectors of covariance matrix scaled by eigenvalue.');


# ### PCA (from Duke course)
# 
# "Principal Components Analysis" (PCA) basically means to find and rank all the eigenvalues and eigenvectors of a covariance matrix. This is useful because high-dimensional data (with $p$ features) may have nearly all their variation in a small number of dimensions $k<p$, i.e. in the subspace spanned by the eigenvectors of the covariance matrix that have the $k$ largest eigenvalues. If we project the original data into this subspace, we can have a dimension reduction (from $p$ to $k$) with hopefully little loss of information.
# 
# Numerically, PCA is typically done using SVD on the data matrix rather than eigendecomposition on the covariance matrix. Numerically, the condition number for working with the covariance matrix directly is the square of the condition number using SVD, so SVD minimizes errors."

# For zero-centered vectors,
# 
# \begin{align}
# \text{Cov}(X, Y) &= \frac{\sum_{i=1}^n(X_i - \bar{X})(Y_i - \bar{Y})}{n-1} \\
#   &= \frac{\sum_{i=1}^nX_iY_i}{n-1} \\
#   &= \frac{XY^T}{n-1}
# \end{align}
# 
# and so the covariance matrix for a data set $X$ that has zero mean in each feature vector is just $XX^T/(n-1)$. 
# 
# In other words, we can also get the eigendecomposition of the covariance matrix from the positive semi-definite matrix $XX^T$.

# Note: Here $x$ is a matrix of **row** vectors.

# In[51]:


X = np.random.random((5,4))
X


# In[52]:


Y = X - X.mean(axis=1)[:, None]  # eliminate the mean
print(Y.mean(axis=1))


# In[53]:


np.around(Y.mean(1), 5)


# In[54]:


Y


# Check that the covariance matrix is unaffected by removing the mean:

# In[55]:


np.cov(X)


# In[56]:


np.cov(Y)


# In[57]:


# Find the eigenvalue and eigenvectors
e1, v1 = np.linalg.eig(np.dot(x, x.T)/(n-1))


# #### Principal components
# 
# Principal components are simply the eigenvectors of the covariance matrix used as basis vectors. Each of the original data points is expressed as a linear combination of the principal components, giving rise to a new set of coordinates. 

# In[58]:


# Check that we reproduce the previous result
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(x[0,:], x[1,:], alpha=0.2)
for e_, v_ in zip(e1, v1.T):
    ax.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
ax.axis([-3,3,-3,3]);
ax.set_aspect(1)


# ### Using SVD for PCA
# 
# SVD is a decomposition of the data matrix $X = U S V^T$ where $U$ and $V$ are orthogonal matrices and $S$ is a diagonal matrix. 
# 
# Recall that the transpose of an orthogonal matrix is also its inverse, so if we multiply on the right by $X^T$, we get the following simplification
# 
# \begin{align}
# X &= U S V^T \\
# X X^T &= U S V^T (U S V^T)^T \\
#  &= U S V^T V S U^T \\
#  &= U S^2 U^T
# \end{align}
# 
# Compare with the eigendecomposition of a matrix $A = W \Lambda W^{-1}$, we see that SVD gives us the eigendecomposition of the matrix $XX^T$, which as we have just seen, is basically a scaled version of the covariance for a data matrix with zero mean, with the eigenvectors given by $U$ and eigenvalues by $S^2$ (scaled by $n-1$)..

# In[59]:


u, s, v = np.linalg.svd(x)


# In[60]:


# reproduce previous results yet again!
e2 = s**2/(n-1)
v2 = u
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(x[0,:], x[1,:], alpha=0.2)
for e_, v_ in zip(e2, v2):
    ax.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
ax.axis([-3,3,-3,3]);
ax.set_aspect(1)


# In[61]:


v1 # from eigenvectors of covariance matrix


# In[ ]:


v2 # from SVD


# In[ ]:


e1 # from eigenvalues of covariance matrix


# In[ ]:


e2 # from SVD


# In[ ]:





# ## Exercises: covariance matrix manipulations in Python (taken from the Duke course)

# Given the following covariance matrix
# ```python
# A = np.array([[2,1],[1,4]])
# ```
# use Python to do these basic tasks (that is, do not do them by hand but use `scipy.linalg` functions).
# 
# 1. Show that the eigenvectors of $A$ are orthogonal. 
# 1. What is the vector representing the first principal component direction? 
# 1. Find $A^{-1}$ without performing a matrix inversion. 
# 1. What are the coordinates of the data points (0, 1) and (1, 1) in the standard basis expressed as coordinates of the principal components? 
# 1. What is the proportion of variance explained if we keep only the projection onto the first principal component? 
# 
# We'll give you a headstart on the Python manipulations (you should take a look at the `scipy.linalg` documentation).

# In[ ]:


A = np.array([[2,1],[1,4]])
eigval, eigvec = la.eig(A)


# In[ ]:





# - Find the matrix $A$ that results in rotating the standard vectors in $\mathbb{R}^2$ by 30 degrees counter-clockwise and stretches $e_1$ by a factor of 3 and contracts $e_2$ by a factor of $0.5$. 
# - What is the inverse of this matrix? How you find the inverse should reflect your understanding.
# 
# The effects of the matrix $A$ and $A^{-1}$ are shown in the figure below:
# 
# ![image](../../_images/vecs.png)

# In[ ]:





# We observe some data points $(x_i, y_i)$, and believe that an appropriate model for the data is that
# 
# $$
# f(x) = ax^2 + bx^3 + c\sin{x}
# $$
# 
# with some added noise. Find optimal values of the parameters $\beta = (a, b, c)$ that minimize $\Vert y - f(x) \Vert^2$
# 
# 1. using `scipy.linalg.lstsq` 
# 2. solving the normal equations $X^TX \beta = X^Ty$ 
# 3. using `scipy.linalg.svd` 
# 
# In each case, plot the data and fitted curve using `matplotlib`.
# 
# Data
# ```
# x = array([ 3.4027718 ,  4.29209002,  5.88176277,  6.3465969 ,  7.21397852,
#         8.26972154, 10.27244608, 10.44703778, 10.79203455, 14.71146298])
# y = array([ 25.54026428,  29.4558919 ,  58.50315846,  70.24957254,
#         90.55155435, 100.56372833,  91.83189927,  90.41536733,
#         90.43103028,  23.0719842 ])
# ```

# In[ ]:


x = np.array([ 3.4027718 ,  4.29209002,  5.88176277,  6.3465969 ,  7.21397852,
        8.26972154, 10.27244608, 10.44703778, 10.79203455, 14.71146298])
y = np.array([ 25.54026428,  29.4558919 ,  58.50315846,  70.24957254,
        90.55155435, 100.56372833,  91.83189927,  90.41536733,
        90.43103028,  23.0719842 ])


# In[ ]:




