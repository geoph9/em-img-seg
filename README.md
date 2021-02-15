## EM Image Segmentation

This is a simple implementation of the EM algorithm for performing image segmentation. The goal is to find the `k` components
that contain the most information about the pixels of an image (a coloured image).

This is done by first converting the 3d array of the image (one extra dimension for each `rgb` color) to a 2d representation.
Then, on top of this new array, we apply the EM algorithm in order to approximate the parameters of the GMM that is defined
for the specified number of clusters/components.

There is a very high possibility that the algorithm is wrong since I actually implemented it 2 years ago without
sufficient knowledge of the topic. The first implementation was in python (the jupyter notebook located under `pure-python`).
Recently, I started learning Rust and so I thought that this would be a good experiment to learn the language.

In my rust implementation, I simply implemented the same formulas without going into much detail. I believe that they
can be simplified since I have used way too many for loops. Also, I am certain that there is a bug in my Rust code
since the output is a bit worse than the one from the python implementation.

I have used `PyO3` for binding my Rust code to a python module (named `rustem`) which can be invoked as `from rustem import EM`.
My Rust code is definitely not the best but I am still learning.

## Contents

- [Usage](#usage)
- [Performance](#performance)
    - [Pure Python Performance](#pure-python-performance)
    - [Python with Rust Performance](#python-with-rust-performance)
- [Installation/Using Locally](#installationusing-locally)
- [The Maths](#the-maths)
    - [Complete Data Likelihood](#complete-data-likelihood)
    - [Expectation Step](#e-step)
    - [Maximization Step](#m-step)
    - [Notes](#notes)

## Usage

Example script:
```python
from rustem import EM, save_new

epochs = 30
n_clusters = 32
img_path = "../images/img.jpg"

# Initialize EM
em = EM(img_path, n_clusters)
# Fit the GMM
em.fit(30)
# Save the new image
save_new(em, "path/to/new/image.jpg")
```
The output can bee seen in the image below:
![k=32 output](images/output_k32.png)

- Left: The initial image.
- Right: The new image after using only 32 colors.

The same image with `k=64` components is:

![k=64 output](images/output_k64.jpg)

As you can see the image is now more improved, and the colors are way less than the initial number.

## Performance

I used [`hyperfine`](https://github.com/sharkdp/hyperfine) for measuring the performance of the pure python implementation
and the python-with-rust implementation.

The hyperparameters used were the following:
- `k=32`: Number of clusters
- `epochs=30`: Number of epochs

### Pure Python Performance
```bash
em-img-seg/pure-python on  master [!+?] via 🐍 v3.9.1 (env)
❯ hyperfine "python em_seg.py"
Benchmark #1: python em_seg.py
  Time (mean ± σ):     50.384 s ±  2.237 s    [User: 132.371 s, System: 88.762 s]
  Range (min … max):   48.214 s … 54.201 s    10 runs
```

So, it took about **50** seconds to run the EM algorithm.

### Python with Rust Performance
```bash
em-img-seg/examples on  master [!+?] via 🐍 v3.9.1 (env)
❯ hyperfine "python segment.py"
Benchmark #1: python segment.py
  Time (mean ± σ):     17.185 s ±  1.443 s    [User: 17.242 s, System: 0.398 s]
  Range (min … max):   15.160 s … 19.588 s    10 runs
```

It took about **17** seconds to run the EM algorithm with the same hyperparameters.

This means that the rust implementation is almost **3 times faster**!

## Installation/Using Locally

You can use `rustem` as a python library after doing the following:
```bash
git clone https://github.com/geoph9/em-img-seg.git
cd em-img-seg/
# Activate or create a new python environment
# python -m venv env
# . ./env/bin/activate
python setup.py install
```
To build the rust library alone, run:
```bash
cargo build
```

If the above ran without any errors then you can now use the `rustem` module as in the example in the [Usage](#usage) section.

## The Maths

We are going to perform image segmentation on a given image input, using the EM algorithm. Our goal will be to maximize the following likelihood:

![equation](https://latex.codecogs.com/gif.latex?p%28x%29%20%3D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%u03C0_k%20%5Cprod_%7Bd%3D1%7D%5E%7BD%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%u03C0%u03C3_k%5E2%7D%7De%5E%7B%5Cfrac%7B1%7D%7B2%u03C3_k%5E2%7D%28x_d%20-%20%u03BC_%7Bkd%7D%29%5E%7B2%7D%7D)

Let's say that `X` is the image array (`width X height X 3)`, `N=width * height` (total number of pixels), `K=n_clusters` and `D=3 (n_colors)`. For the GMM, we also have the following:

![equation](https://latex.codecogs.com/gif.latex?%5Cmu%3A%20Mean%20%5Cnewline%20%5Csigma%3A%20CovMatrix%20%5Cnewline%20%5Cpi%3A%20Priors)

(Note that I am using as `sigma` a diagonal covariance matrix and so I am simply representing it as a 1d array.)

### Complete Data Likelihood:

Let's say that we denote the latent variable that describes our complete data as `zeta`. The complete data likelihood (before applying the log function ) will look like this:

![equation](https://latex.codecogs.com/gif.latex?p%28X%2C%20Z%7C%5Cmu%2C%20%5Csigma%5E2%2C%20%5Cpi%29%20%3D%20%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20%5Cprod_%7Bk%3D1%7D%5E%7BK%7D%20%5B%5Cpi_k%20%5Cprod_%7Bd%3D1%7D%5E%7BD%7D%20%5Cmu_%7Bkd%7D%5E%7BX_%7Bnd%7D%7D%20%281-%5Cmu_%7Bkd%7D%29%5E%7B1-X_%7Bnd%7D%7D%5D%5E%7BZ_%7Bnk%7D%7D)

### E-Step

Now, we will compute the **Expected Complete Log Likelihood**. Since each Zeta appears linearly in complete data log likelihood then we can approximate zeta with the average of that (gamma).

We are going to calculate gamma which will be a `NxK` matrix that will show us if `X_n` is likely to belong to cluster `K`. Since we have a mixture of Gaussian distributions, then:

![equation](https://latex.codecogs.com/gif.latex?%5Cgamma%28z_%7Bnk%7D%29%20%3D%20%5Cfrac%7B%5Cpi_k%20N%28X_n%7C%5Cmu_k%2C%20%5Csigma_k%5E2%29%7D%7B%5Csum_%20%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_j%20N%28X_n%7C%5Cmu_j%2C%20%5Csigma_j%5E2%29%7D)

So, if we apply the $log$ function we have:

![equation](https://latex.codecogs.com/gif.latex?%5Cgamma%28z_%7Bnk%7D%29%20%3D%20%5Cfrac%7Be%5E%7B%5Csum_%20%7Bk%3D1%7D%5E%7BK%7D%20%5Clog%7B%5Cpi_k%7D%20-%20%5Csum_%20%7Bd%3D1%7D%5E%7BD%7D%20%5Clog%7B%5Csqrt%7B2%5Cpi%20%5Csigma_k%5E2%7D%7D%20&plus;%20%5Cfrac%7B%28x_d%20-%20%5Cmu_%7Bkd%7D%29%5E%7B2%7D%7D%7B2%5Csigma_k%5E2%7D%7D%7D%20%7B%5Csum_%20%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_j%20N%28X_n%7C%5Cmu_j%2C%20%5Csigma_j%5E2%29%7D)

### M-Step

Now, we should tune the model parameters in order to maximize the likelihood. This step is the most important since it defines how our model will react.

In order to get the new values for our parameters we will maximize the **Expected Complete Log Likelihood** we saw earlier.
Since we have 3 parameters, then we will maximize the function for each one. To do that we have to calculate the derivatives with respect to each of these parameters (`mu`, `sigma`, `pi`).
The output will be the following:

1.  For `mu` we will simply use an average in order to find which pixels belong to cluster `K`.
    Since we have `D=3` colors then `mu` will be calculated as such:
    ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_%7Bkd%7D%20%3D%20%5Cfrac%7B%5Csum_%20%7Bn%3D1%7D%5E%7BN%7D%20%5Cgamma%28z_%7Bnk%7D%29X_nd%7D%7B%5Csum_%20%7Bn%3D1%7D%5E%7BN%7D%20%5Cgamma%28z_%7Bnk%7D%29%7D)
2.  For `sigma` we only have `K` positions. That means that we will have to group the information about the color to
    just one cell (for each k value). The updated sigma values will be:
    ![equation](https://latex.codecogs.com/gif.latex?%5Csigma_k%5E2%20%3D%20%5Cfrac%7B1%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cgamma_%7Bnk%7D%7D%20%5Csum_%20%7Bn%3D1%7D%5E%7BN%7D%20%5Csum_%20%7Bd%3D1%7D%5E%7BD%7D%20%5Cgamma_%7Bnk%7D%28X_n%20-%20%5Cmu_%7Bkd%7D%29%5E2)
3.  At last, we need to update the apriori distribution for the next step. It can be easily proved that `pi` always has
    the same form, no matter the initial distribution (so if we have a mixture of Poisson distributions, the `pi` would
    be updated in the same way):
    ![equation](https://latex.codecogs.com/gif.latex?%5Cpi_k%20%3D%20%5Cfrac%7B%5Csum_%20%7Bn%3D1%7D%5E%7BN%7D%20%5Cgamma%28z_%7Bnk%7D%29%7D%7BN%7D)

### Notes
The above was taken from the jupyter notebook located under the `pure-python` directory. The formulas have been copy-pasted and converted to latex using
[Latex Codecogs](https://www.codecogs.com/latex/eqneditor.php).
