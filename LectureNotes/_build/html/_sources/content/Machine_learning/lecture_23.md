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
        * Plot with `plot_acquisition`.

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

## Bayesian neutral networks (BNNs): What is the idea?

Think of how we might combine Bayesian ideas with neural networks (NNs) or modify NNs.

* We use BNNs when we care about uncertainty.

* Standard NN training via optimization is equivalent to doing maximum likelihood estimation (MLE) for the weights.
    * So point estimates only.
    * Recall issues with MLEs from "Why Bayes is Better" section.
    * General problem for NNs: susceptible to overfitting.

* Can address overfitting (in part) by *regularization* $\Lra$ don't let weights get too big.
    * Bayesian equivalent: put priors on weights as we used naturalness priors in Mini-project I (an L2 regularization $\Lra$ Gaussian prior pdf).
    * So now finding a MAP estimate (maximizing posterior).

* New research on overfitting: in some cases a learning algorithm makes good predictions despite fitting to the noise in training data. This is called "benign overfitting". See, for example, [Bartlett et al., *Benign Overfitting in Linear Regression*](https://arxiv.org/abs/1906.11300). A recent explanation of this is [Chatterji and Long, *Foolish Crowds Support Benign Overfitting*](https://arxiv.org/abs/2110.02914). From their abstract: "Our analysis exposes the benefit of an effect analogous to the 'wisdom of the crowd', except here the harm arising from fitting the *noise* is ameliorated by spreading it among many directions - the variance reduction arises from a *foolish* crowd."

* Bayesian way: posterior inference $\Lra$ BNNs (start with model, update with data).
    * This is a challenge both to model and to compute.
    * Approximations such as Laplace's method are inadequate and MCMC is computationally infeasible (because of too many parameters).
    * The alternative is *variational inference*, where one approximates the posterior.
    * Important for decision-making systems and smaller data situations.

* Ordinary workflow of a neural network (supervised learning)
    * Randomly initialize the weights.
    * Given inputs, compute outputs of neurons by layers, propagate to prediction.
    * A "loss function" computes deviation of predicted output $\hat y$ and expected $y$.
    * The loss value is "back propagated" through layers, adjusting the weights.
* Output of ordinary ML does not come with variability or a credibility
    * Just a point prediction - no model of the world is explicitly constructed.
    * There are weights and network topology, but no direct correlation to a statistical model.

* BNN: Prior describes key parameters, utilized as input to the neural net. Output is used to compute likelihood with pdf. Get the posterior distribution by variational inference.

## Notebooks from Christian Forssen's course at Chalmers

A. [](/notebooks/Machine_learning/Bayesian_neural_networks_tif285.ipynb)

B. [](/notebooks/Machine_learning/demo-Bayesian_neural_networks_tif285.ipynb) 

Some notes from A:
* Basic neural network. Bottom line goal is

    $$
     p(y|\xvec, D) = \int p(y|\xvec,\wvec) p(\wvec|D) d\wvec
    $$  

    where $y$ is the new output, $\xvec$ is the new input$, and $D = \{\xvec^{(i)},y^{(i)}\}$ is a given training dataset.
    * The first pdf in the integral is what the neural network gives $\Lra$ deterministic given $\xvec$ and $\wvec$.
    * We marginalize over the weights.

* We need $p(\wvec|D) \propto p(D|\wvec)p(\wvec)$ by Bayes.
Then $p(D|\wvec) = \prod_i p(y^{(i)}| \xvec^{(i)},\wvec)$ is the likelihood.

* So how do we calculate the marginalization integral with thousands of parameters? $\Lra$ *variational inference* or VI.

* Quick idea about VI and KL divergence.
    * $p(\wvec|D)$ is intractable so approximate the true posterior with a proxy variational distribution $q(\wvec|\thetavec)$, where we need to find the optimal $\thetavec$, denoted $\thetavec^\ast$.
        * Find $\thetavec^\ast$ by using the Kullback-Leibler divergence.
        * Find $\thetavec^\ast$ that maximizes $J_{\rm ELBD}(\thetavec)$, where ELBD stands for "Evidence Lower Bound".


