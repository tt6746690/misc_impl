#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Reference:
#     gpflow: https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
#             https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py#L263
#     julia:  https://github.com/STOR-i/GaussianProcesses.jl/blob/master/src/sparse/fully_indep_train_conditional.jl
#     ladax:  https://github.com/danieljtait/ladax
#

import sys
sys.path.append('../kernel')

import numpy as np
import numpy.random as npr
np.set_printoptions(precision=3,suppress=True)

import jax
from jax import device_put
import jax.numpy as np
import jax.numpy.linalg as linalg

from typing import Any, Callable, Sequence, Optional, Tuple
import flax
from flax import linen as nn
from flax import optim

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
print(torch.cuda.is_available(), jax.devices())

import itertools

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')

from tabulate import tabulate

import sys
sys.path.append('../kernel')
from jaxkern import (cov_se, cov_rq, cov_pe, LookupKernel, normalize_K, mtgp_k)

from plt_utils import plt_savefig, plt_scaled_colobar_ax
from gp import gp_regression_chol, run_sgd
from gpax import log_func_default, log_func_simple, flax_run_optim, CovSE, GPR, GPRFITC, is_psd


# In[3]:



## Parameters 

xlim = (-1, 1)
ylim = (-2, 2)
n_train = 200
n_test = 200
σn = .5
logsn = np.log(σn)
lr = .01
num_steps = 20


## Data

def f_gen(x):
    return np.sin(x * 3 * 3.14) +            0.3 * np.cos(x * 9 * 3.14) +            0.5 * np.sin(x * 7 * 3.14)

## Plotting

key = jax.random.PRNGKey(0)
npr.seed(0)
X_train = np.expand_dims(npr.uniform(xlim[0], xlim[1], size=n_train), 1)
y_train = f_gen(X_train) + σn * npr.rand(n_train, 1)
data = (X_train, y_train)
X_test  = np.expand_dims(np.linspace(*xlim, n_test), 1)

X_train = device_put(X_train)
y_train = device_put(y_train)
X_test = device_put(X_test)

print(X_train.shape, y_train.shape)


fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(X_train, y_train, 'x', alpha=1)
ax.grid()
ax.set_ylim(ylim)


# In[37]:



from jax.scipy.linalg import cho_solve


class GPRFITC(nn.Module):
    data: Tuple[np.ndarray, np.ndarray]
    n_inducing: int

    def setup(self):
        self.k = CovSE()
        self.logσn = self.param('logσn',
                                nn.initializers.zeros, (1,))
        X, y = self.data
        self.Xu = self.param('Xu', lambda k,s : X[:self.n_inducing],
                             (self.n_inducing, X.shape[-1]))

    def get_init_params(self, key):
        params = self.init(key, np.ones((1, self.data[0].shape[-1])),
                           method=self.pred_f)
        return params
    
    def precompute(self):
        X, y = self.data
        k = self.k
        σ2 = np.exp(2*self.logσn)
        Xu = self.Xu
        n, m = len(X), self.n_inducing
        
        Kdiag = k(X, diag=True)
        Kuu = k(Xu, Xu)
        Kuf = k(Xu, X)
        Luu = linalg.cholesky(Kuu+1e-6*np.eye(m))
        
        V = cho_solve((Luu, True), Kuf)
        Qff = V.T@V
        Λ = Kdiag - np.diag(Qff) + σ2
        
        B = np.eye(m) + (V/Λ)@V.T
        LB = linalg.cholesky(B)
        γ = cho_solve((LB, True), (V/Λ)@y)

        return Luu, Λ, LB, γ
    
    def mll(self):
        X, y = self.data
        n = len(X)
        Luu, Λ, LB, γ = self.precompute()
        
        mll_quad  = -.5*( sum(y/Λ*y) - sum(np.square(γ)) )[0]
        mll_det   = -np.sum(np.log(np.diag(LB)))-.5*np.sum(np.log(Λ))
        mll_const = -(n/2)*np.log(2*np.pi)
        mll = mll_quad + mll_det + mll_const

        return mll
    
    def pred_f(self, Xs):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        n = len(X)
        Luu, Λ, LB, γ = self.precompute()
        
        Kss = k(Xs, Xs)
        Kus = k(Xu, Xs)
        ω = cho_solve((Luu, True), Kus)
        ν = cho_solve((LB, True), ω)
        
        μ = ω.T@linalg.solve(LB.T, γ)
        Σ = Kss - ω.T@ω + ν.T@ν
        
        return μ, Σ

    def pred_y(self, Xs):
        σ2 = np.exp(2*self.logσn)
        μf, Σf = self.pred_f(Xs)
        ns = len(Σf)
        μy, Σy = μf, Σf + σ2*np.diag(np.ones((ns,)))
        return μy, Σy
    
    

model = GPRFITC(data, 30)
params = model.get_init_params(key)
# model.apply(params, method=model.mll)

k = CovSE()
Xu = params['params']['Xu']
Kuu = k.apply({'params': params['params']['k']}, Xu, Xu)

np.all(linalg.eigvals(Kuu+1e-6*np.eye(len(Kuu))) > 0)

# Luu = np.linalg.cholesky(Kuu+1e-6*np.eye(len(Kuu)))
# Luu

# b = np.ones((len(Xu), 1))

# import jax.scipy as jscipy


# A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
# b = np.array([1, 1, 1, 1])
# L, low = jscipy.linalg.cho_factor(A)
# L = np.linalg.cholesky(A)
# low = True
# x = jscipy.linalg.cho_solve((L, low), b)
# np.allclose(A @ x - b, np.zeros(4), atol=1e-5)


# x = linalg.solve(A, b)
# np.allclose(A @ x - np.array([1, 1, 1, 1]), np.zeros(4), atol=1e-5)





# jscipy.linalg.cho_solve((Luu, True), b), linalg.solve(Luu, b)
# (A @ x - np.array([1, 1, 1, 1]))*100000
# jax.scipy.linalg.cho_solve






# In[ ]:




optimizer_kwargs = {'learning_rate': .001}
num_steps = 200
n_inducing = 50


fig, axs = plt.subplots(1, 2, figsize=(20, 5), sharey=True)

def get_model(i):
    if i == 0:
        return 'GPR', GPR(data)
    if i == 1:
        return 'GPR+FITC', GPRFITC(data, n_inducing)
    
    
def log_func(i, f, params):
#     if i%10==0:
    print(f'[{i:3}]\tLoss={f(params):.3f}\t'
          f'σn={np.exp(params["params"]["logσn"][0]):.3f}')

for i in range(2):
    if i == 0:
        continue
    name, model = get_model(i)
    params = model.get_init_params(key)
    nmll = lambda params: -model.apply(params, method=model.mll)
#     nmll = jax.jit(nmll)
    params = flax_run_optim(nmll, params, num_steps=num_steps, log_func=log_func,
                            optimizer_kwargs=optimizer_kwargs)
    
    mll = model.apply(params, method=model.mll)
    μ, Σ = model.apply(params, X_test, method=model.pred_y)
    std = np.expand_dims(np.sqrt(np.diag(Σ)), 1)

    ax = axs[i]
    ax.plot(X_test, μ, color='k')
    ax.fill_between(X_test.squeeze(), (μ-2*std).squeeze(), (μ+2*std).squeeze(), alpha=.2, color=cmap(0))
    ax.scatter(X_train, y_train, marker='x', color='r', s=50, alpha=.4)
    if i == 1:
        Xu = params['params']['Xu']
        ax.plot(Xu, np.zeros_like(Xu), "k|", mew=2, label="Inducing locations")
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f'{name}: mll={-mll:.2f}')
    


# In[ ]:




