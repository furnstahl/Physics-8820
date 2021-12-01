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
     p(\omega|D,\alpha) = \frac{p(D|\wvec) p(\wvec|\alpha)}{p(D|\alpha)}.
    $$

    * $p(D|\wvec) = e^{-C(\wvec)}$ tells us that $C(\wvec)$ is minus the log likelihood. So $p(D|\wvec)$ is the product of the (assumed independent) probabilities for each input-output pair.

    * The normalized prior is

        $$
         p(\wvec|\alpha) = \frac{1}{Z_W(\alpha)} e^{-\alpha E_W}
        $$

        so $\alpha E_W$ is minus the log prior pdf.

     