# Lecture 23

## Step through Mini-project IIIa to see what is needed

Here is a summary of what you need to do for Mini-project IIIa:

* Code for function and a standard `scipy` optimization.
* Specification of statistical model and acquisition function, plotting 10 (or so) iteration and summary plots.
* Change acquisition function to 'LCB'. What's that?
* Assessing exploration vs. exploitation.
* Adding noise.
* Changing the GP kernel and analyzing the results.
* Optional task: implementing BayesOpt algorithm (open the black box!)
* Bivariate example **for a plus** $\Lra$ follow details. 

Now go back to [](/notebooks/Machine_learning/Bayesian_optimization) and see where these elements come from.

1. Ingredients:

    * Objective function to optimize 
        $f(\thetavec) = \chi^2(\thetavec) = \sum_{i=1}^N [y_i^{\text{expt}} - y_i^{\text{th}(\thetavec)}]^2/\sigma_i^2$, which is assumed to be costly to evaluate.
        **Find $\thetavec_\ast$ that minimizes $\chi^2$.

    * Statistical model for $f(\thetavec)$: $p(f | \mathcal{D})$ where $\mathcal{D}$ is the current set of evaluations of $f$:
    $\mathcal{D} = \{\thetavec_i, f(\thetavec_i)\}$. Start with $p(f)$ as a Gaussian process GP and *update* via Bayes theorem (recall the GP procedure) with each additional $(\thetavec_i,f(\thetavec_i))$.

    * Acquisition function $\mathcal{A}(\thetavec|\mathcal{D})$. Take maximum with respect to $\thetavec$ to determine $\theta_{i+1}$, given the current data $\mathcal{D}$ (the full history). This balances between "exploration" and "exploitation".

2. Step through the code

    * Univariate example: plot function and find minimum from `scipy.optimize.minimum` (note: requires a starting point).
        * no noise at first

    * Create GPyOpt object with `GPyOpt.methods.BayesianOptimization`.
        * specify objective function $f(\thetavec)$, domain, initial data, acquisition function, whether or not exact function.

    * `run_optimization(max_iter, max_time, eps)`, where `max_iter` is the number of evaluations, `max_time` is the time budget, and `eps` is the minimum distance between $\thetavec_i$ and $\thetavec_{i+1}$.
        * Plot with `plot_acquition`.

    * Bivariate example
        * This uses a built-in example.
        * Notice the setup for the BayesianOptimization oject
            * look at choices for `model_type`
            * look at choices for `acquisition_type`
        * separate plots for mean, standard deviation (sd), acquisition function, with red dots for evaluation points.

3. Look at options for starting samples
    * LHS and Mersenne twister (standard rng) have "holes" but the LHS projections are good.
    * Sobol sequence does not have hole.

4. Concluding remarks: step through these.

5. Step back to acquisition function
    
    * Expective improvement $\Lra$ EI. At each $\thetavec$ point, calculation the expectation value of $f_{\rm min} - f(\thetavec)$, where $f_{\rm min}$ is the lowest result so far.
    This is the improvement; use 0 if not improvement ($f_{\rm min} - f(\thetavec) > 0$).

    * Analytic evaluation of expectation value, because at a given $\theta_\ast$ the distribution pdf for $f$ is a Gaussian (because it is a GP) specfied by $\mu(\thetavec_\ast)$ and $\sigma^2(\thetavec_\ast) = C(\thetavec_\ast,\thetavec_\ast)$.

    * Consider the two pieces to the integral with $z = (f_{\rm min} - \mu)/\sigma$:
        * *explorative* if $\phi(z)$ dominates $\Lra$ prior has large uncertainty (large $\sigma$);
        * *exploitive* if $z\Phi(z)$ dominates $\Lra$ prior has low $\mu$.
        * With LCB you can change the relative importance of these two terms.

## Bayesian neutral networks: What is the idea?
