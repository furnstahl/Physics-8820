# Lecture 24

## Brief notes on Neural network classifier demonstration

These are some brief comments on the notebook [](/notebooks/Machine_learning/Forssen_tif285_demo-NeuralNet.ipynb), which demonstrates a neural network classifier using TensorFlow.

* TensorFlow is input with: `import tensorflow as tf`. A simple environment file suited to run this notebook is `environment_ml.yml`:
    ```
    name: 8820-env-ml
    channels:
    - conda-forge
    dependencies:
    - python=3.8
    - numpy
    - scipy
    - matplotlib
    - pandas
    - seaborn
    - jupyter
    - nbdime
    - pip
    - pip:
      - tensorflow
      - keras
      - tqdm
    ```
    * In principle we could get tensorflow, keras, and tqdm from conda, but that seems to have some problems that we avoid by using `pip`.
* `Keras` is used to build the model neural network to classify images of handwritten numbers (0 to 9), which we train and test.
* The data set is the [MNIST database](http://yann.lecun.com/exdb/mnist/), which is a standard test case with 60000 training examples and 10000 test examples.
* Try plotting different entries in `x_train`, e.g., `x_train[301]`. This is a $60000 \times 28 \times 28$ array, so specifying just the first number gives a $28\times 28$ array. This is the basic input information to the neural net. The array `y_train` has the correct numbers (as integers); check that it agrees with the picture.
* If you do `np.min(x_train)` and `np.max(x_train)` you'll find that the values go from 0 to 255, so we divide `x_train` and `x_test` by 255 so it is scaled from 0 to 1.
* The network will have a series of layers.
    * `tf.keras.models.Sequential` says that it is a sequential model, which just means a series of layers.
    * `tf.keras.layers.Flatten(input_shape=(28, 28))` takes the $28\times 28$ array and flattens it into a single vector of length 784, because we will need a vector of inputs (here each a number between 0 and 1).
    * We make the hidden layer with `tf.keras.layers.Dense(128, activation='relu')` where `Dense` means to connect all the neurons, and the activation function is `relu`. Do `shift+tab+tab` to see other options. See the [keras manual](https://keras.io/api/layers/activations/) for other options such as `sigmoid`, `softmax`, `tanh`, and `selu`.
    * Next is `tf.keras.layers.Dropout(0.2)`, which is a layer that sets some fraction (here 0.2) of inputs to 0, to help prevent overfitting. (Why does this help?)
    * The final layer is `  tf.keras.layers.Dense(10, activation='softmax')`, which gives 10 output probabilities that sum to one. Each node is the probability for that integer being the one represented in the input picture.
* Before training we compile the model with an optimizer that  (here `adam` instead of stochastic gradient descent, see [this page](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)); a loss function (`loss`) that is to be minimized (see [options here](https://www.tensorflow.org/api_docs/python/tf/keras/losses)); and `metrics`, which is chosen here to be `accuracy`, which is the fraction of correctly classified images (which it will know from `y_train`).
* Train with `model.fit(x_train, y_train, epochs=5)`, where an epoch is an iteration over the entire `x` and `y` data provided.
    * Note how the accuracy improves. 
    * Would more epochs help?
* Now we test with   
`test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)`
* Look at individual predictions: note the probabilities. Do they make sense from your intuition about what mistake could be made?
* Play with the plot to see where there are failures.
* Try adding another layer and compare to doubling the number of nodes from 128. Try removing the dropout layer.

## Stepping through a Bayesian neural network notebook

We return to [](/notebooks/Machine_learning/Bayesian_neural_networks_tif285.ipynb).
Note that Mini-project-IIIb involves playing with and interpreting [](/notebooks/Machine_learning/demo-Bayesian_neural_networks_tif285.ipynb).

Basic neural network:
* Each neuron has input $\xvec = \{x_i\}, i=1,\ldots I$, that is, a vector of $I$ numbers, which has an output $y$, which depends *nonlinearly* on the activation $a$:

    $$
     a = w_0 + \sum_{i=1}^I w_i x_i
    $$ 

    with $\wvec = \{w_i\}, i=1,\ldots I$ the weights to be determined by training. The activation function could be a sigmoid or a tanh function or something else (as we saw above).

* We identify a *loss function* for the problem; training means feed the network with training data and adjust the weights to minimize the loss. So this is just like a generalization of doing a least-squares fit to determine the parameters of a fitting function like a polynomial. The difference here is that the functions that can be fit are completely general.

* Classification here means that the single output $y$ is a real probabilitiy (i.e., $y \in [0,1]$) that the input $\xvec$ belong to one of two classes, $t=1$ or $t=0$. (Looking ahead, $\xvec$ has the coordinates of the datapoint and the $t$ values map to blue or red.) So

    $$\begin{align}
       y &= p(t=1|\wvec,\xvec) \equiv p_{t=1} \\
       1-y &= p(t=0|\wvec,\xvec) \equiv p_{t=0} .
    \end{align}$$  

* The loss function chosen for this problem and minimized with respect to $\wvec$ is

    $$
     C_W(\wvec) = C(\wvec) + \alpha E_W(\wvec) ,
    $$

    where the "error function" is

    $$
     C(\wvec) = -\sum_n \Bigl[ t^{(n)} \log\bigl(y(\xvec^{(n)},\wvec)\bigr) +
       (1 - t^{(n)}) \log\bigl(1-y(\xvec^{(n)},\wvec))\bigr)\Bigr] .
    $$

    * The sum over $n$ is over the training data set $\{\xvec^{(n)}, t^{(n)}\}$.

    * So the first log contributes if $t^{(n)} = 1$ and the second log contributes if $t^{(n)}=0$.

    * Check that this means that $C(\wvec)$ is zero if the classification is 100% correct. 

* The other term $E_W(\wvec) = \frac{1}{2}\sum_i w_i^2$ is a *regularizer*, which inhibits overfitting by penalizing (i.e., yielding a large loss) overly large weights. If the weights can get large, there can be fine cancellations to fit fluctuations in the data; this is overfitting.

* We can interpret this loss function in terms of a familiar Bayesian statistical model for finding the weights give data $D$ and a value for $\alpha$:

    $$
     p(\wvec|D,\alpha) = \frac{p(D|\wvec) p(\wvec|\alpha)}{p(D|\alpha)}.
    $$

    * $p(D|\wvec) = e^{-C(\wvec)}$ tells us that $C(\wvec)$ is minus the log likelihood. So $p(D|\wvec)$ is the product of the (assumed independent) probabilities for each input-output pair.

    * The normalized prior is

        $$
         p(\wvec|\alpha) = \frac{1}{Z_W(\alpha)} e^{-\alpha E_W}
        $$

        so $\alpha E_W$ is minus the log prior pdf.
        If $E_W$ is quadratic, this is a Gaussian with variance $\sigma_W^2 = 1/\alpha$ and normalization $Z_W(\alpha) = (2\pi/\alpha)^{I+1}$. Then

        $$
           p(\wvec|D, \alpha) = \frac{1}{Z_M(\alpha)} e^{-[C(\wvec) + \alpha E_W(\wvec)]} = \frac{1}{Z_M(\alpha)} e^{-[C_W(\wvec)} . 
        $$

* The figure in the notebook at this point shows some training data and the resulting posteriors.
    * What do you observe?
    * $N=0$ is just the prior, so what is $\alpha$?
    * Note the separation of red and blue data.

* The machine learning approach is often to minimize $C_W(\wvec)$ to find the MAP $\wvec^*$. The Bayesian approach is to consider the information in the actual pdf (rather than the mode only).

* Notation: $y$ is output from the neural network. For classification problems (like here), $y$ is the discrete categorial distributions of probabilities $p_{t=c}$ for class $c$. For regresssion (fitting a function), $y$ is continuous. If $y$ is a vector, then the network creates a nonlinear map $y(\xvec,\wvec): x \in \mathbb{R}^p \rightarrow y \in \mathbb{R}^m$. 

:::{admonition} Classification of uncertainties:  
* Epistemic - uncertainties in the model, so can be reduced with more data.
* Aleatoric - from noise in the training data. E.g., Gaussian noise. More of the same data doesn't change the noise.
:::

### Probabilistic model

Recap: the goal of a Bayesian neural network (BNN) is to infer

$$
 p(y|\xvec, D) = \int p(y|\xvec,\wvec) p(\wvec|D) d\wvec
$$  

where $y$ is the new output, $\xvec$ is the new input, and $D = \{\xvec^{(i)},y^{(i)}\}$ is a given training dataset.

* The first pdf in the integral is what the neural network gives $\Lra$ deterministic given $\xvec$ and $\wvec$.
* We marginalize over the weights.

* We need $p(\wvec|D) \propto p(D|\wvec)p(\wvec)$ by Bayes.
* Then $p(D|\wvec) = \prod_i p(y^{(i)}| \xvec^{(i)},\wvec)$ is the likelihood (because of independence).

* As before, $p(\wvec)$ helps prevent overfitting by *regularizing* the weights.

* Calculating the marginalization integral over $\wvec$ is the same as averaging the predictions from an ensemble of NNs weighted by the posterior probabilities of their weights given the data (i.e., by $p(\wvec|D)$).

* Now back to the binary ($t=1$ or 0) classification problem. The marginalization integral for the $(n+1)^{\rm th}$ point is:

    $$
     p(y^{(n+1)} | \xvec^{(n+1)},D,\alpha) =
     \int p(y^{(n+1)} | \xvec^{(n+1)},\wvec,\alpha) p(\wvec|D,\alpha)\,     d\wvec .
    $$

    We could also marginalize over $\alpha$.

* The figures in the notebook show the classification in Bayesian (left panel) and regular (optimization) form. Here $\xvec = (x_1,x_2)$ (incorrect $y$-axis label; $x_1$ should be $x_2$) and $\alpha = 1.0$. 
    * The decision boundary is $y = 0.5$, which is activation $a=0$ and $a = \pm 1,\pm 2$ lines are also shown. There correspond to $y = 0.12, 0.27, 0.73,, 0.88$. The test data are pluses while the training data are circles.
    * I don't know why the colors are reversed in the right banel.

* The Bayesian results are from sampling many neurons with different weights, distributed proportional to the posterior pdf. The decision boundary is from the mean of the sample predictions evaluated on a grid.
    * The next figure plots the *standard deviation* of the predictions.


* Recap of methods for the marginalization integral
    1. Sampling, e.g., by MCMC.
    2. Analytic approximations, e.g., Gaussian approximation to Laplace method.
    3. Variational method. This will be the method of choice for large numbers of parameters (as in ML applications).

### Variational inference for Bayesian neural networks

* The basic idea is to approximate the true posterior by a parametrized posterior and adjust the parameters $\thetavec$ to optimize the agreement. Therefore optimize rather than sample.
    * Use $q(\wvec|\thetavec)$ to approximate $p(\wvec|D)$, using $\theta = \theta^*$ as the optimal values.
    * Use the Kullback-Leibler (KL) divergence as a measure for how close we are (here for notational simplicity $q(\wvec|\thetavec) \rightarrow q(\wvec)$ and $p(\wvec|D) \rightarrow p(\wvec)$):

    $$
      D_{KL}(q \Vert p) = \int q(\wvec)\log\frac{q(\wvec)}{p(\wvec)} d\wvec
      \equiv \mathbb{E}_{\qvec}[\log q(\wvec) - \log p(\wvec)]      
    $$

    where the expectation value $\mathbb{E}_{\qvec}$ is with respect to the pdf $q(\wvec)$.

* The variational property is that this quantity is $\geq 0$ and only equal to zero if $q(\wvec) = p(\wvec)$, with the "best" approximation when this is minimized.

* We can prove that  $D_{KL}(q \Vert p) \geq 0$ several ways. One of the easiest is to use that

    $$ 
       \log x \leq x - 1 \quad\mbox{for }x>0 .
    $$ 

    * Try graphing it!
    * You can show this by demonstrating that $x \leq e^{x-1}$, considering the cases $x<1$ and $x>1$ separately. (Hint: change variables in each case and use the Taylor expansion of $e^z$.)
    * Given this result, we have (be careful of numerator and denominator in the logarithm and the resulting sign)

    $$\begin{align}
      - D_{KL}(q \Vert p) &= \int q(\wvec)\log\frac{p(\wvec)}{q(\wvec)} d\wvec \\
         &\leq \int q(\wvec)\bigl(\frac{p(\wvec)}{q(\wvec)} - 1\bigr) d\wvec \\
         &= \int p(\wvec)\,d\wvec - \int q(\wvec)\,d\wvec = 1 - 1 = 0
    \end{align}$$

    so $D_{KL}(q \Vert p) \geq 0$.
    
    * In the second line we use $p(\wvec),q(\wvec) \geq 0$ but handle $p(\wvec)$ and $q(\wvec) = 0$ separately.     
    \end{align}

* The KL-divergence that is sometimes seen in this context is $D_{KL}(p \Vert q) \neq D_{KL}(q \Vert p)$, but both have the vaiational feature.
    * We favor $D_{KL}(q \Vert p)$ here because the $q(\wvec)$ distribution is known for taking expectation values.
    * Note that minimizing  

    $$\begin{align}
     D_{KL}(q \Vert p) &= \int q(\wvec|\thetavec)\log\frac{q(\wvec|\thetavec)}{p(\wvec|D)} d\wvec \\
     & = -\int q(\wvec|\thetavec)\log p(\wvec|D)\, d\wvec +
     \int q(\wvec|\thetavec)\log q(\wvec|\thetavec)\,d\wvec
    \end{align}$$

    where the first term requires implausible parameters to be avoided to minimize while in the second term one maximizes the entropy of the variational distribution $q$.

#### Evidence Lower Bound (ELBO)

* The ELBO appears when we apply Bayes theorem to $p(\wvec|D)$ and substitute in to $D_{KL}(q \Vert p)$:

    $$\begin{align}
      D_{KL}(q \Vert p) &= \int q(\wvec|\thetavec)
      [\log q(\wvec|\thetavec) - \log p(D|\wvec) - \log p(\wvec) + \log p(D)] \\
      &= \mathbb{E}_q[\log p(\wvec|\thetavec)] -\mathbb{E}_q[\log p(D|\wvec)] - \mathbb{E}_q[\log p(\wvec)] + \log p(D)
    \end{align}$$

    where $\log p(D)$ is independent of $\wvec$ and we have used the normalization of $q$.

* Because $D_{KL}(q \Vert p) \geq 0$, we have

    $$
      \log p(D) \geq -\mathbb{E}_q[\log q(\wvec|\thetavec)]
      + \mathbb{E}_q[\log p(D|\wvec)]  +\mathbb{E}_q[\log p(\wvec)]
      \equiv J_{ELBO}(\thetavec) .
    $$

* $-J_{ELBO}(\thetavec)$ is also called the variational free energy $F(D,\thetavec)$.

* Goal: find $\thetavec^*$ that maximizes $J_{ELBO}(\thetavec)$.
    * The hardest term is $\mathbb{E}_q[\log p(D|\wvec)]$.
    * Recall that $p(D|\wvec) = \Pi_i p(y^{(i)}|\xvec^{(i)},\wvec)$ so $\mathbb{E}_q[\log p(D|\wvec)] = \sum_{i=1}^{N} \mathbb{E}_q[\log p(y^{(i)}|\xvec^{(i)},\wvec)]$.
* There is active research in improving how to find $\thetavec^*$. 
