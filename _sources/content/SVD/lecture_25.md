# Lecture 25

We'll do a brief introduction to singular value decomposition via a Jupyter notebook: [](/notebooks/SVD/linear_algebra_games_including_SVD.ipynb). We won't be able to look at everything (or prove everything) in detail, so we will start with the highlights and fill in what we can.

## Preliminary exercises

* Linear algebra using the index form. (This is duplicated from [Lecture 13](/content/Why_Bayes_is_better/lecture_13.md).)
Recall that

    $$
      (AB)^\intercal = B^\intercal A^\intercal
      \quad\mbox{and}\quad
     (A^\intercal)_{ji} = A_{ij}
    $$

    $$
      \chi^2 = [Y - A\thetavec]^\intercal\, \Sigmavec^{-1}\, [Y - A\thetavec] =
      (Y_i - A_{ij}\thetavec_j)(\Sigma^{-1})_{ii'}(Y_{i'}- A_{i'j'}\thetavec_{j'}) ,
    $$

    where $i,i'$ run from $1$ to $N$ and $j,j'$ run from one to $p$, where the highest power term is $x^{p-1}$. 
    * Summed over $i,i',j,j'$ because they each appear (exactly) twice.
    * $(\Sigma^{-1})_{ii'} \neq (\Sigma_{i,i'})^{-1}$
    * Be sure you understand the indices on the leftmost term, remembering that the matrix expression has this term transposed.
    * We know $\chi^2$ is a scalar because there are no free indices.

* We find the MLE from $\partial\chi^2/\partial\thetavec_k = 0$ for $k = 1,\ldots p$. 

    $$\begin{align}
     \left.\frac{\partial\chi^2}{\partial\thetavec_k}\right|_    {\thetavec=\thetavechat}
     &= -A_{ij}\delta_{jk}(\Sigma^{-1})_{i,i'}(Y_{i'} - A_    {i'j'}\thetavechat_{j'}) + 
     (Y_{i} - A_{ij}\thetavechat_{j})(\Sigma^{-1})_{i,i'}(-A_    {i'j'}\delta_{j'k}) = 0
    \end{align}$$

* Isolate the $\thetavec$ terms on one side and show the doubled terms are equal:

    $$
     A_{ik}(\Sigmavec^{-1})_{i,i'}Y_{i'}
     + Y_i (\Sigmavec^{-1})_{i,i'} A_{i'k}
     =
     A_{ik}(\Sigmavec^{-1})_{i,i'}A_{i'j'}\thetavechat_{j'}
     + A_{ij}\thetavechat_{j}(\Sigmavec^{-1})_{i,i'}A_{i'k}
    $$   

    * In the second term on the left, switch $i\leftrightarrow i'$ and use $(\Sigmavec^{-1})_{i',i} = (\Sigmavec^{-1})_{i,i'}$ because it is symmetric. This is then the same as the first term.
    * In the first term on the right, we switch $j\leftrightarrow j'$ and use the symmetry of $\Sigmavec$ again to show the two terms are the same. 

* Writing $A_{ik} = (A^\intercal)_{ki}$, we get
    
    $$\begin{align}
     2(A^\intercal)_{ki} (\Sigmavec^{-1})_{i,i'} Y_i
      = 2 (A^{\intercal})_{ki} (\Sigmavec^{-1})_{i,i'} A_    {i'j}\thetavechat_j
    \end{align}$$
    
    or, removing the indices,
    
    $$
      (A^{\intercal}\Sigmavec^{-1} Y) = (A^{\intercal}\Sigmavec^{-1}A)    \thetavechat
    $$
    
    and then inverting (which is possible because the expression in parentheses on the right is a square, invertible matrix), we finally obtain:
    
    $$
      \thetavechat = [A^\intercal \Sigmavec^{-1} A]^{-1}
         [A^\intercal \Sigmavec^{-1} Y] .
    $$

    Q.E.D.



## Singular value decomposition (SVD)

* The main point of SVD is the decomposition of a matrix into three other matrices.
    * We can do full SVD or reduced SVD. The former is usually the starting point but the latter is used in practice.

    * Reduced form is $ A = U S V^\intercal$, with $S$ *diagonal*. 
    * $A$ is an $m\times n$ matrix, while $U$ is $m \times n$, $S$ is $n\times n$, and $V$ is $n\times n$.

    * Elements of $S$, names $S_{kk'} \equiv S_k \delta_{kk'}$  are *singular values*. For a square matrix $A$, they would be the eigenvalues.

    * Key feature:

        $$
           A_{ij} \approx \sum_{k=1}^p U_{ik} S_k V_{jk} \ \mbox{with } p < n    
        $$
    
        is a truncated representation of $A$ with most of the content if the "dominant" $p$ vectors kept, which are identified by the largest singular values.


## Applications of SVD

* Solving matrix equations with *ill-conditioned matrices*, which means that the smallest eigenvalue is zero or close to zero. (If zero, then the matrix is *singular*.)
    * The *condition number* of a matrix is the ratio of the largest to the smallest eigenvalue.
    * If solving $Ax = h$ $\Lra$ $x = A^{-1}b$, then an error in $h$ is magnified by the condition number to give the error in $x$.
    * So finite precision in $b$ leads to nonsense for $x$ if the the condition number is larger than the inverse of the machine or calculational precision. If the condition number $\kappa(A) = 10^k$, then up to $k$ digits of accuracy is lost (roughly).

* Data reduction application: identify the *important* basis elements $\Lra$ e.g., what linear combinations of parameters are most important.
    * Note the Python syntax for matrix multiplication with `@`, transpose with `T`.
    * Step through the Python example for image compression.
    You are identifying the most important eigenvalues for the image.
    * If we keep 1 in 100 of the singulare values, then the numbers to be stored are significantly reduced.
    * Note the spectrum of eigenvalues.

* Covariance, PCA, and SVD
    * Consider the covariance matrix and find its eigenvalues and eigenvectors. ``Center'' the data first (subtract the mean so that you deal with data having mean zero).
    * PCA means to rank these. This is typically done *via* SVD (which is better numerically).
    * Then one uses fewer singular values (which are the eigenvalues if it is a square matrix) $\Lra$ we have reduced the number of parameters in a model (for example).


