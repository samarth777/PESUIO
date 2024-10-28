Title: 3.1. Linear Regression — Dive into Deep Learning 1.0.3 documentation

URL Source: https://d2l.ai/chapter_linear-regression/linear-regression.html

Markdown Content:
3.1. Linear Regression[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#linear-regression "Permalink to this heading")

Open the notebook in Colab

[](https://studiolab.sagemaker.aws/import/github/d2l-ai/d2l-pytorch-sagemaker-studio-lab/blob/main/GettingStarted-D2L.ipynb)Open the notebook in SageMaker Studio Lab


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_Regression_ problems pop up whenever we want to predict a numerical value. Common examples include predicting prices (of homes, stocks, etc.), predicting the length of stay (for patients in the hospital), forecasting demand (for retail sales), among numerous others. Not every prediction problem is one of classical regression. Later on, we will introduce classification problems, where the goal is to predict membership among a set of categories.

As a running example, suppose that we wish to estimate the prices of houses (in dollars) based on their area (in square feet) and age (in years). To develop a model for predicting house prices, we need to get our hands on data, including the sales price, area, and age for each home. In the terminology of machine learning, the dataset is called a _training dataset_ or _training set_, and each row (containing the data corresponding to one sale) is called an _example_ (or _data point_, _instance_, _sample_). The thing we are trying to predict (price) is called a _label_ (or _target_). The variables (age and area) upon which the predictions are based are called _features_ (or _covariates_).

%matplotlib inline
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

![Image 1: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

%matplotlib inline
import math
import time
from mxnet import np
from d2l import mxnet as d2l

![Image 2: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

%matplotlib inline
import math
import time
from jax import numpy as jnp
from d2l import jax as d2l

![Image 3: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

No GPU/TPU found, falling back to CPU. (Set TF\_CPP\_MIN\_LOG\_LEVEL\=0 and rerun for more info.)

%matplotlib inline
import math
import time
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

![Image 4: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

3.1.1. Basics[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#basics "Permalink to this heading")
--------------------------------------------------------------------------------------------------------------------

_Linear regression_ is both the simplest and most popular among the standard tools for tackling regression problems. Dating back to the dawn of the 19th century ([Gauss, 1809](https://d2l.ai/chapter_references/zreferences.html#id80 "Gauss, C. F. (1809). Theoria motus corporum coelestum. Werke. Königlich Preussische Akademie der Wissenschaften."), [Legendre, 1805](https://d2l.ai/chapter_references/zreferences.html#id164 "Legendre, A. M. (1805). Mémoire sur les Opérations Trigonométriques: dont les Résultats Dépendent de la Figure de la Terre. F. Didot.")), linear regression flows from a few simple assumptions. First, we assume that the relationship between features x and target y is approximately linear, i.e., that the conditional mean E\[Y∣X\=x\] can be expressed as a weighted sum of the features x. This setup allows that the target value may still deviate from its expected value on account of observation noise. Next, we can impose the assumption that any such noise is well behaved, following a Gaussian distribution. Typically, we will use n to denote the number of examples in our dataset. We use superscripts to enumerate samples and targets, and subscripts to index coordinates. More concretely, x(i) denotes the ith sample and xj(i) denotes its jth coordinate.

### 3.1.1.1. Model[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#model "Permalink to this heading")

At the heart of every solution is a model that describes how features can be transformed into an estimate of the target. The assumption of linearity means that the expected value of the target (price) can be expressed as a weighted sum of the features (area and age):

(3.1.1)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-eq-price-area "Permalink to this equation")

price\=warea⋅area+wage⋅age+b.

Here warea and wage are called _weights_, and b is called a _bias_ (or _offset_ or _intercept_). The weights determine the influence of each feature on our prediction. The bias determines the value of the estimate when all features are zero. Even though we will never see any newly-built homes with precisely zero area, we still need the bias because it allows us to express all linear functions of our features (rather than restricting us to lines that pass through the origin). Strictly speaking, [(3.1.1)](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-eq-price-area) is an _affine transformation_ of input features, which is characterized by a _linear transformation_ of features via a weighted sum, combined with a _translation_ via the added bias. Given a dataset, our goal is to choose the weights w and the bias b that, on average, make our model’s predictions fit the true prices observed in the data as closely as possible.

In disciplines where it is common to focus on datasets with just a few features, explicitly expressing models long-form, as in [(3.1.1)](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-eq-price-area), is common. In machine learning, we usually work with high-dimensional datasets, where it is more convenient to employ compact linear algebra notation. When our inputs consist of d features, we can assign each an index (between 1 and d) and express our prediction y^ (in general the “hat” symbol denotes an estimate) as

(3.1.2)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-0 "Permalink to this equation")

y^\=w1x1+⋯+wdxd+b.

Collecting all features into a vector x∈Rd and all weights into a vector w∈Rd, we can express our model compactly via the dot product between w and x:

In [(3.1.3)](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-eq-linreg-y), the vector x corresponds to the features of a single example. We will often find it convenient to refer to features of our entire dataset of n examples via the _design matrix_ X∈Rn×d. Here, X contains one row for every example and one column for every feature. For a collection of features X, the predictions y^∈Rn can be expressed via the matrix–vector product:

where broadcasting ([Section 2.1.4](https://d2l.ai/chapter_preliminaries/ndarray.html#subsec-broadcasting)) is applied during the summation. Given features of a training dataset X and corresponding (known) labels y, the goal of linear regression is to find the weight vector w and the bias term b such that, given features of a new data example sampled from the same distribution as X, the new example’s label will (in expectation) be predicted with the smallest error.

Even if we believe that the best model for predicting y given x is linear, we would not expect to find a real-world dataset of n examples where y(i) exactly equals w⊤x(i)+b for all 1≤i≤n. For example, whatever instruments we use to observe the features X and labels y, there might be a small amount of measurement error. Thus, even when we are confident that the underlying relationship is linear, we will incorporate a noise term to account for such errors.

Before we can go about searching for the best _parameters_ (or _model parameters_) w and b, we will need two more things: (i) a measure of the quality of some given model; and (ii) a procedure for updating the model to improve its quality.

### 3.1.1.2. Loss Function[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#loss-function "Permalink to this heading")

Naturally, fitting our model to the data requires that we agree on some measure of _fitness_ (or, equivalently, of _unfitness_). _Loss functions_ quantify the distance between the _real_ and _predicted_ values of the target. The loss will usually be a nonnegative number where smaller values are better and perfect predictions incur a loss of 0. For regression problems, the most common loss function is the squared error. When our prediction for an example i is y^(i) and the corresponding true label is y(i), the _squared error_ is given by:

(3.1.5)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-eq-mse "Permalink to this equation")

l(i)(w,b)\=12(y^(i)−y(i))2.

The constant 12 makes no real difference but proves to be notationally convenient, since it cancels out when we take the derivative of the loss. Because the training dataset is given to us, and thus is out of our control, the empirical error is only a function of the model parameters. In [Fig. 3.1.1](https://d2l.ai/chapter_linear-regression/linear-regression.html#fig-fit-linreg), we visualize the fit of a linear regression model in a problem with one-dimensional inputs.

![Image 5: ../_images/fit-linreg.svg](https://d2l.ai/_images/fit-linreg.svg)

Fig. 3.1.1 Fitting a linear regression model to one-dimensional data.[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#id10 "Permalink to this image")

Note that large differences between estimates y^(i) and targets y(i) lead to even larger contributions to the loss, due to its quadratic form (this quadraticity can be a double-edge sword; while it encourages the model to avoid large errors it can also lead to excessive sensitivity to anomalous data). To measure the quality of a model on the entire dataset of n examples, we simply average (or equivalently, sum) the losses on the training set:

(3.1.6)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-1 "Permalink to this equation")

L(w,b)\=1n∑i\=1nl(i)(w,b)\=1n∑i\=1n12(w⊤x(i)+b−y(i))2.

When training the model, we seek parameters (w∗,b∗) that minimize the total loss across all training examples:

(3.1.7)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-2 "Permalink to this equation")

w∗,b∗\=argminw,b L(w,b).

### 3.1.1.3. Analytic Solution[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#analytic-solution "Permalink to this heading")

Unlike most of the models that we will cover, linear regression presents us with a surprisingly easy optimization problem. In particular, we can find the optimal parameters (as assessed on the training data) analytically by applying a simple formula as follows. First, we can subsume the bias b into the parameter w by appending a column to the design matrix consisting of all 1s. Then our prediction problem is to minimize ‖y−Xw‖2. As long as the design matrix X has full rank (no feature is linearly dependent on the others), then there will be just one critical point on the loss surface and it corresponds to the minimum of the loss over the entire domain. Taking the derivative of the loss with respect to w and setting it equal to zero yields:

(3.1.8)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-3 "Permalink to this equation")

∂w‖y−Xw‖2\=2X⊤(Xw−y)\=0 and hence X⊤y\=X⊤Xw.

Solving for w provides us with the optimal solution for the optimization problem. Note that this solution

will only be unique when the matrix X⊤X is invertible, i.e., when the columns of the design matrix are linearly independent ([Golub and Van Loan, 1996](https://d2l.ai/chapter_references/zreferences.html#id89 "Golub, G. H., & Van Loan, C. F. (1996). Matrix Computations. Johns Hopkins University Press.")).

While simple problems like linear regression may admit analytic solutions, you should not get used to such good fortune. Although analytic solutions allow for nice mathematical analysis, the requirement of an analytic solution is so restrictive that it would exclude almost all exciting aspects of deep learning.

### 3.1.1.4. Minibatch Stochastic Gradient Descent[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#minibatch-stochastic-gradient-descent "Permalink to this heading")

Fortunately, even in cases where we cannot solve the models analytically, we can still often train models effectively in practice. Moreover, for many tasks, those hard-to-optimize models turn out to be so much better that figuring out how to train them ends up being well worth the trouble.

The key technique for optimizing nearly every deep learning model, and which we will call upon throughout this book, consists of iteratively reducing the error by updating the parameters in the direction that incrementally lowers the loss function. This algorithm is called _gradient descent_.

The most naive application of gradient descent consists of taking the derivative of the loss function, which is an average of the losses computed on every single example in the dataset. In practice, this can be extremely slow: we must pass over the entire dataset before making a single update, even if the update steps might be very powerful ([Liu and Nocedal, 1989](https://d2l.ai/chapter_references/zreferences.html#id176 "Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45(1), 503–528.")). Even worse, if there is a lot of redundancy in the training data, the benefit of a full update is limited.

The other extreme is to consider only a single example at a time and to take update steps based on one observation at a time. The resulting algorithm, _stochastic gradient descent_ (SGD) can be an effective strategy ([Bottou, 2010](https://d2l.ai/chapter_references/zreferences.html#id23 "Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. Proceedings of COMPSTAT'2010 (pp. 177–186). Springer.")), even for large datasets. Unfortunately, SGD has drawbacks, both computational and statistical. One problem arises from the fact that processors are a lot faster multiplying and adding numbers than they are at moving data from main memory to processor cache. It is up to an order of magnitude more efficient to perform a matrix–vector multiplication than a corresponding number of vector–vector operations. This means that it can take a lot longer to process one sample at a time compared to a full batch. A second problem is that some of the layers, such as batch normalization (to be described in [Section 8.5](https://d2l.ai/chapter_convolutional-modern/batch-norm.html#sec-batch-norm)), only work well when we have access to more than one observation at a time.

The solution to both problems is to pick an intermediate strategy: rather than taking a full batch or only a single sample at a time, we take a _minibatch_ of observations ([Li _et al._, 2014](https://d2l.ai/chapter_references/zreferences.html#id167 "Li, M., Zhang, T., Chen, Y., & Smola, A. J. (2014). Efficient mini-batch training for stochastic optimization. Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 661–670).")). The specific choice of the size of the said minibatch depends on many factors, such as the amount of memory, the number of accelerators, the choice of layers, and the total dataset size. Despite all that, a number between 32 and 256, preferably a multiple of a large power of 2, is a good start. This leads us to _minibatch stochastic gradient descent_.

In its most basic form, in each iteration t, we first randomly sample a minibatch Bt consisting of a fixed number |B| of training examples. We then compute the derivative (gradient) of the average loss on the minibatch with respect to the model parameters. Finally, we multiply the gradient by a predetermined small positive value η, called the _learning rate_, and subtract the resulting term from the current parameter values. We can express the update as follows:

(3.1.10)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-5 "Permalink to this equation")

(w,b)←(w,b)−η|B|∑i∈Bt∂(w,b)l(i)(w,b).

In summary, minibatch SGD proceeds as follows: (i) initialize the values of the model parameters, typically at random; (ii) iteratively sample random minibatches from the data, updating the parameters in the direction of the negative gradient. For quadratic losses and affine transformations, this has a closed-form expansion:

(3.1.11)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-eq-linreg-batch-update "Permalink to this equation")

w←w−η|B|∑i∈Bt∂wl(i)(w,b)\=w−η|B|∑i∈Btx(i)(w⊤x(i)+b−y(i))b←b−η|B|∑i∈Bt∂bl(i)(w,b)\=b−η|B|∑i∈Bt(w⊤x(i)+b−y(i)).

Since we pick a minibatch B we need to normalize by its size |B|. Frequently minibatch size and learning rate are user-defined. Such tunable parameters that are not updated in the training loop are called _hyperparameters_. They can be tuned automatically by a number of techniques, such as Bayesian optimization ([Frazier, 2018](https://d2l.ai/chapter_references/zreferences.html#id72 "Frazier, P. I. (2018). A tutorial on Bayesian optimization. ArXiv:1807.02811.")). In the end, the quality of the solution is typically assessed on a separate _validation dataset_ (or _validation set_).

After training for some predetermined number of iterations (or until some other stopping criterion is met), we record the estimated model parameters, denoted w^,b^. Note that even if our function is truly linear and noiseless, these parameters will not be the exact minimizers of the loss, nor even deterministic. Although the algorithm converges slowly towards the minimizers it typically will not find them exactly in a finite number of steps. Moreover, the minibatches B used for updating the parameters are chosen at random. This breaks determinism.

Linear regression happens to be a learning problem with a global minimum (whenever X is full rank, or equivalently, whenever X⊤X is invertible). However, the loss surfaces for deep networks contain many saddle points and minima. Fortunately, we typically do not care about finding an exact set of parameters but merely any set of parameters that leads to accurate predictions (and thus low loss). In practice, deep learning practitioners seldom struggle to find parameters that minimize the loss _on training sets_ ([Frankle and Carbin, 2018](https://d2l.ai/chapter_references/zreferences.html#id71 "Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: finding sparse, trainable neural networks. ArXiv:1803.03635."), [Izmailov _et al._, 2018](https://d2l.ai/chapter_references/zreferences.html#id134 "Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. ArXiv:1803.05407.")). The more formidable task is to find parameters that lead to accurate predictions on previously unseen data, a challenge called _generalization_. We return to these topics throughout the book.

### 3.1.1.5. Predictions[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#predictions "Permalink to this heading")

Given the model w^⊤x+b^, we can now make _predictions_ for a new example, e.g., predicting the sales price of a previously unseen house given its area x1 and age x2. Deep learning practitioners have taken to calling the prediction phase _inference_ but this is a bit of a misnomer—_inference_ refers broadly to any conclusion reached on the basis of evidence, including both the values of the parameters and the likely label for an unseen instance. If anything, in the statistics literature _inference_ more often denotes parameter inference and this overloading of terminology creates unnecessary confusion when deep learning practitioners talk to statisticians. In the following we will stick to _prediction_ whenever possible.

3.1.2. Vectorization for Speed[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#vectorization-for-speed "Permalink to this heading")
------------------------------------------------------------------------------------------------------------------------------------------------------

When training our models, we typically want to process whole minibatches of examples simultaneously. Doing this efficiently requires that we vectorize the calculations and leverage fast linear algebra libraries rather than writing costly for-loops in Python.

To see why this matters so much, let’s consider two methods for adding vectors. To start, we instantiate two 10,000-dimensional vectors containing all 1s. In the first method, we loop over the vectors with a Python for-loop. In the second, we rely on a single call to `+`.

n \= 10000
a \= torch.ones(n)
b \= torch.ones(n)

![Image 6: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

n \= 10000
a \= np.ones(n)
b \= np.ones(n)

![Image 7: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

\[22:06:52\] ../src/storage/storage.cc:196: Using Pooled (Naive) StorageManager for CPU

n \= 10000
a \= jnp.ones(n)
b \= jnp.ones(n)

![Image 8: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

n \= 10000
a \= tf.ones(n)
b \= tf.ones(n)

![Image 9: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

Now we can benchmark the workloads. First, we add them, one coordinate at a time, using a for-loop.

c \= torch.zeros(n)
t \= time.time()
for i in range(n):
    c\[i\] \= a\[i\] + b\[i\]
f'{time.time() \- t:.5f} sec'

![Image 10: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

c \= np.zeros(n)
t \= time.time()
for i in range(n):
    c\[i\] \= a\[i\] + b\[i\]
f'{time.time() \- t:.5f} sec'

![Image 11: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

\# JAX arrays are immutable, meaning that once created their contents
\# cannot be changed. For updating individual elements, JAX provides
\# an indexed update syntax that returns an updated copy
c \= jnp.zeros(n)
t \= time.time()
for i in range(n):
    c \= c.at\[i\].set(a\[i\] + b\[i\])
f'{time.time() \- t:.5f} sec'

![Image 12: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

c \= tf.Variable(tf.zeros(n))
t \= time.time()
for i in range(n):
    c\[i\].assign(a\[i\] + b\[i\])
f'{time.time() \- t:.5f} sec'

![Image 13: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

Alternatively, we rely on the reloaded `+` operator to compute the elementwise sum.

t \= time.time()
d \= a + b
f'{time.time() \- t:.5f} sec'

![Image 14: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

t \= time.time()
d \= a + b
f'{time.time() \- t:.5f} sec'

![Image 15: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

t \= time.time()
d \= a + b
f'{time.time() \- t:.5f} sec'

![Image 16: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

t \= time.time()
d \= a + b
f'{time.time() \- t:.5f} sec'

![Image 17: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

The second method is dramatically faster than the first. Vectorizing code often yields order-of-magnitude speedups. Moreover, we push more of the mathematics to the library so we do not have to write as many calculations ourselves, reducing the potential for errors and increasing portability of the code.

3.1.3. The Normal Distribution and Squared Loss[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#the-normal-distribution-and-squared-loss "Permalink to this heading")
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

So far we have given a fairly functional motivation of the squared loss objective: the optimal parameters return the conditional expectation E\[Y∣X\] whenever the underlying pattern is truly linear, and the loss assigns large penalties for outliers. We can also provide a more formal motivation for the squared loss objective by making probabilistic assumptions about the distribution of noise.

Linear regression was invented at the turn of the 19th century. While it has long been debated whether Gauss or Legendre first thought up the idea, it was Gauss who also discovered the normal distribution (also called the _Gaussian_). It turns out that the normal distribution and linear regression with squared loss share a deeper connection than common parentage.

To begin, recall that a normal distribution with mean μ and variance σ2 (standard deviation σ) is given as

(3.1.12)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-6 "Permalink to this equation")

p(x)\=12πσ2exp⁡(−12σ2(x−μ)2).

Below we define a function to compute the normal distribution.

def normal(x, mu, sigma):
    p \= 1 / math.sqrt(2 \* math.pi \* sigma\*\*2)
    return p \* np.exp(\-0.5 \* (x \- mu)\*\*2 / sigma\*\*2)

![Image 18: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

def normal(x, mu, sigma):
    p \= 1 / math.sqrt(2 \* math.pi \* sigma\*\*2)
    return p \* np.exp(\-0.5 \* (x \- mu)\*\*2 / sigma\*\*2)

![Image 19: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

def normal(x, mu, sigma):
    p \= 1 / math.sqrt(2 \* math.pi \* sigma\*\*2)
    return p \* jnp.exp(\-0.5 \* (x \- mu)\*\*2 / sigma\*\*2)

![Image 20: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

def normal(x, mu, sigma):
    p \= 1 / math.sqrt(2 \* math.pi \* sigma\*\*2)
    return p \* np.exp(\-0.5 \* (x \- mu)\*\*2 / sigma\*\*2)

![Image 21: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

We can now visualize the normal distributions.

\# Use NumPy again for visualization
x \= np.arange(\-7, 7, 0.01)

\# Mean and standard deviation pairs
params \= \[(0, 1), (0, 2), (3, 1)\]
d2l.plot(x, \[normal(x, mu, sigma) for mu, sigma in params\], xlabel\='x',
         ylabel\='p(x)', figsize\=(4.5, 2.5),
         legend\=\[f'mean {mu}, std {sigma}' for mu, sigma in params\])

![Image 22: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

![Image 23: ../_images/output_linear-regression_d0729f_78_0.svg](https://d2l.ai/_images/output_linear-regression_d0729f_78_0.svg)

\# Use NumPy again for visualization
x \= np.arange(\-7, 7, 0.01)

\# Mean and standard deviation pairs
params \= \[(0, 1), (0, 2), (3, 1)\]
d2l.plot(x.asnumpy(), \[normal(x, mu, sigma).asnumpy() for mu, sigma in params\], xlabel\='x',
         ylabel\='p(x)', figsize\=(4.5, 2.5),
         legend\=\[f'mean {mu}, std {sigma}' for mu, sigma in params\])

![Image 24: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

![Image 25: ../_images/output_linear-regression_d0729f_81_0.svg](https://d2l.ai/_images/output_linear-regression_d0729f_81_0.svg)

\# Use JAX NumPy for visualization
x \= jnp.arange(\-7, 7, 0.01)
\# Mean and standard deviation pairs
params \= \[(0, 1), (0, 2), (3, 1)\]
d2l.plot(x, \[normal(x, mu, sigma) for mu, sigma in params\], xlabel\='x',
         ylabel\='p(x)', figsize\=(4.5, 2.5),
         legend\=\[f'mean {mu}, std {sigma}' for mu, sigma in params\])

![Image 26: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

![Image 27: ../_images/output_linear-regression_d0729f_84_0.svg](https://d2l.ai/_images/output_linear-regression_d0729f_84_0.svg)

\# Use NumPy again for visualization
x \= np.arange(\-7, 7, 0.01)

\# Mean and standard deviation pairs
params \= \[(0, 1), (0, 2), (3, 1)\]
d2l.plot(x, \[normal(x, mu, sigma) for mu, sigma in params\], xlabel\='x',
         ylabel\='p(x)', figsize\=(4.5, 2.5),
         legend\=\[f'mean {mu}, std {sigma}' for mu, sigma in params\])

![Image 28: Copy to clipboard](https://raw.githubusercontent.com/choldgraf/sphinx-copybutton/master/sphinx_copybutton/_static/copy-button.svg)

![Image 29: ../_images/output_linear-regression_d0729f_87_0.svg](https://d2l.ai/_images/output_linear-regression_d0729f_87_0.svg)

Note that changing the mean corresponds to a shift along the x\-axis, and increasing the variance spreads the distribution out, lowering its peak.

One way to motivate linear regression with squared loss is to assume that observations arise from noisy measurements, where the noise ϵ follows the normal distribution N(0,σ2):

(3.1.13)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-7 "Permalink to this equation")

y\=w⊤x+b+ϵ where ϵ∼N(0,σ2).

Thus, we can now write out the _likelihood_ of seeing a particular y for a given x via

(3.1.14)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-8 "Permalink to this equation")

P(y∣x)\=12πσ2exp⁡(−12σ2(y−w⊤x−b)2).

As such, the likelihood factorizes. According to _the principle of maximum likelihood_, the best values of parameters w and b are those that maximize the _likelihood_ of the entire dataset:

(3.1.15)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-9 "Permalink to this equation")

P(y∣X)\=∏i\=1np(y(i)∣x(i)).

The equality follows since all pairs (x(i),y(i)) were drawn independently of each other. Estimators chosen according to the principle of maximum likelihood are called _maximum likelihood estimators_. While, maximizing the product of many exponential functions, might look difficult, we can simplify things significantly, without changing the objective, by maximizing the logarithm of the likelihood instead. For historical reasons, optimizations are more often expressed as minimization rather than maximization. So, without changing anything, we can _minimize_ the _negative log-likelihood_, which we can express as follows:

(3.1.16)[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#equation-chapter-linear-regression-linear-regression-10 "Permalink to this equation")

−log⁡P(y∣X)\=∑i\=1n12log⁡(2πσ2)+12σ2(y(i)−w⊤x(i)−b)2.

If we assume that σ is fixed, we can ignore the first term, because it does not depend on w or b. The second term is identical to the squared error loss introduced earlier, except for the multiplicative constant 1σ2. Fortunately, the solution does not depend on σ either. It follows that minimizing the mean squared error is equivalent to the maximum likelihood estimation of a linear model under the assumption of additive Gaussian noise.

3.1.4. Linear Regression as a Neural Network[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#linear-regression-as-a-neural-network "Permalink to this heading")
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

While linear models are not sufficiently rich to express the many complicated networks that we will introduce in this book, (artificial) neural networks are rich enough to subsume linear models as networks in which every feature is represented by an input neuron, all of which are connected directly to the output.

[Fig. 3.1.2](https://d2l.ai/chapter_linear-regression/linear-regression.html#fig-single-neuron) depicts linear regression as a neural network. The diagram highlights the connectivity pattern, such as how each input is connected to the output, but not the specific values taken by the weights or biases.

![Image 30: ../_images/singleneuron.svg](https://d2l.ai/_images/singleneuron.svg)

Fig. 3.1.2 Linear regression is a single-layer neural network.[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#id11 "Permalink to this image")

The inputs are x1,…,xd. We refer to d as the _number of inputs_ or the _feature dimensionality_ in the input layer. The output of the network is o1. Because we are just trying to predict a single numerical value, we have only one output neuron. Note that the input values are all _given_. There is just a single _computed_ neuron. In summary, we can think of linear regression as a single-layer fully connected neural network. We will encounter networks with far more layers in later chapters.

### 3.1.4.1. Biology[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#biology "Permalink to this heading")

Because linear regression predates computational neuroscience, it might seem anachronistic to describe linear regression in terms of neural networks. Nonetheless, they were a natural place to start when the cyberneticists and neurophysiologists Warren McCulloch and Walter Pitts began to develop models of artificial neurons. Consider the cartoonish picture of a biological neuron in [Fig. 3.1.3](https://d2l.ai/chapter_linear-regression/linear-regression.html#fig-neuron), consisting of _dendrites_ (input terminals), the _nucleus_ (CPU), the _axon_ (output wire), and the _axon terminals_ (output terminals), enabling connections to other neurons via _synapses_.

![Image 31: ../_images/neuron.svg](https://d2l.ai/_images/neuron.svg)

Fig. 3.1.3 The real neuron (source: “Anatomy and Physiology” by the US National Cancer Institute’s Surveillance, Epidemiology and End Results (SEER) Program).[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#id12 "Permalink to this image")

Information xi arriving from other neurons (or environmental sensors) is received in the dendrites. In particular, that information is weighted by _synaptic weights_ wi, determining the effect of the inputs, e.g., activation or inhibition via the product xiwi. The weighted inputs arriving from multiple sources are aggregated in the nucleus as a weighted sum y\=∑ixiwi+b, possibly subject to some nonlinear postprocessing via a function σ(y). This information is then sent via the axon to the axon terminals, where it reaches its destination (e.g., an actuator such as a muscle) or it is fed into another neuron via its dendrites.

Certainly, the high-level idea that many such units could be combined, provided they have the correct connectivity and learning algorithm, to produce far more interesting and complex behavior than any one neuron alone could express arises from our study of real biological neural systems. At the same time, most research in deep learning today draws inspiration from a much wider source. We invoke Russell and Norvig ([2016](https://d2l.ai/chapter_references/zreferences.html#id244 "Russell, S. J., & Norvig, P. (2016). Artificial Intelligence: A Modern Approach. Pearson Education Limited.")) who pointed out that although airplanes might have been _inspired_ by birds, ornithology has not been the primary driver of aeronautics innovation for some centuries. Likewise, inspiration in deep learning these days comes in equal or greater measure from mathematics, linguistics, psychology, statistics, computer science, and many other fields.

3.1.5. Summary[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#summary "Permalink to this heading")
----------------------------------------------------------------------------------------------------------------------

In this section, we introduced traditional linear regression, where the parameters of a linear function are chosen to minimize squared loss on the training set. We also motivated this choice of objective both via some practical considerations and through an interpretation of linear regression as maximimum likelihood estimation under an assumption of linearity and Gaussian noise. After discussing both computational considerations and connections to statistics, we showed how such linear models could be expressed as simple neural networks where the inputs are directly wired to the output(s). While we will soon move past linear models altogether, they are sufficient to introduce most of the components that all of our models require: parametric forms, differentiable objectives, optimization via minibatch stochastic gradient descent, and ultimately, evaluation on previously unseen data.

3.1.6. Exercises[¶](https://d2l.ai/chapter_linear-regression/linear-regression.html#exercises "Permalink to this heading")
--------------------------------------------------------------------------------------------------------------------------

1.  Assume that we have some data x1,…,xn∈R. Our goal is to find a constant b such that ∑i(xi−b)2 is minimized.
    
    1.  Find an analytic solution for the optimal value of b.
        
    2.  How does this problem and its solution relate to the normal distribution?
        
    3.  What if we change the loss from ∑i(xi−b)2 to ∑i|xi−b|? Can you find the optimal solution for b?
        
2.  Prove that the affine functions that can be expressed by x⊤w+b are equivalent to linear functions on (x,1).
    
3.  Assume that you want to find quadratic functions of x, i.e., f(x)\=b+∑iwixi+∑j≤iwijxixj. How would you formulate this in a deep network?
    
4.  Recall that one of the conditions for the linear regression problem to be solvable was that the design matrix X⊤X has full rank.
    
    1.  What happens if this is not the case?
        
    2.  How could you fix it? What happens if you add a small amount of coordinate-wise independent Gaussian noise to all entries of X?
        
    3.  What is the expected value of the design matrix X⊤X in this case?
        
    4.  What happens with stochastic gradient descent when X⊤X does not have full rank?
        
5.  Assume that the noise model governing the additive noise ϵ is the exponential distribution. That is, p(ϵ)\=12exp⁡(−|ϵ|).
    
    1.  Write out the negative log-likelihood of the data under the model −log⁡P(y∣X).
        
    2.  Can you find a closed form solution?
        
    3.  Suggest a minibatch stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint: what happens near the stationary point as we keep on updating the parameters)? Can you fix this?
        
6.  Assume that we want to design a neural network with two layers by composing two linear layers. That is, the output of the first layer becomes the input of the second layer. Why would such a naive composition not work?
    
7.  What happens if you want to use regression for realistic price estimation of houses or stock prices?
    
    1.  Show that the additive Gaussian noise assumption is not appropriate. Hint: can we have negative prices? What about fluctuations?
        
    2.  Why would regression to the logarithm of the price be much better, i.e., y\=log⁡price?
        
    3.  What do you need to worry about when dealing with pennystock, i.e., stock with very low prices? Hint: can you trade at all possible prices? Why is this a bigger problem for cheap stock? For more information review the celebrated Black–Scholes model for option pricing ([Black and Scholes, 1973](https://d2l.ai/chapter_references/zreferences.html#id19 "Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81, 637–654.")).
        
8.  Suppose we want to use regression to estimate the _number_ of apples sold in a grocery store.
    
    1.  What are the problems with a Gaussian additive noise model? Hint: you are selling apples, not oil.
        
    2.  The [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) captures distributions over counts. It is given by p(k∣λ)\=λke−λ/k!. Here λ is the rate function and k is the number of events you see. Prove that λ is the expected value of counts k.
        
    3.  Design a loss function associated with the Poisson distribution.
        
    4.  Design a loss function for estimating log⁡λ instead.
