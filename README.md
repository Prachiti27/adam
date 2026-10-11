# Adam: A Method for Stochastic Optimization

A simple, dependency-light NumPy implementation of **Adam** and **AdaMax** from the paper:

> Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.
> [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy
```

## Quick start

Run the demo, which trains logistic regression on synthetic, badly scaled data with each optimizer:

```bash
python adam.py
```


## Key ideas from the paper

- **Adaptive per-parameter step sizes** from estimates of the first and second moments of the gradient.
- **Bias correction** fixes the zero-initialization bias of the moving averages, which matters most when `beta2` is close to 1.
- **Bounded steps:** each update is roughly bounded by `lr`, so the learning rate is easy to choose.
- **Scale invariant:** rescaling the gradients does not change the update.
- **AdaMax** is a variant based on the infinity norm, with a simpler update bound and no bias correction for the second term.

