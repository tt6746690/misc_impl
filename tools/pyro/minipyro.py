# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mini Pyro
---------

This file contains a minimal implementation of the Pyro Probabilistic
Programming Language. The API (method signatures, etc.) match that of
the full implementation as closely as possible. This file is independent
of the rest of Pyro, with the exception of the :mod:`pyro.distributions`
module.

An accompanying example that makes use of this implementation can be
found at examples/minipyro.py.
"""
import random
import warnings
import weakref
from collections import OrderedDict

import torch

from pyro.distributions import validation_enabled

# Pyro keeps track of two kinds of global state:
# i)  The effect handler stack, which enables non-standard interpretations of
#     Pyro primitives like sample();
#     See http://docs.pyro.ai/en/stable/poutine.html
# ii) Trainable parameters in the Pyro ParamStore;
#     See http://docs.pyro.ai/en/stable/parameters.html

PYRO_STACK = []
PARAM_STORE = {}  # maps name -> (unconstrained_value, constraint)


def get_param_store():
    return PARAM_STORE


# The base effect handler class (called Messenger here for consistency with Pyro).
class Messenger:
    def __init__(self, fn=None):
        self.fn = fn

    # Effect handlers push themselves onto the PYRO_STACK.
    # Handlers earlier in the PYRO_STACK are applied first.
    def __enter__(self):
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


# A first useful example of an effect handler.
# trace records the inputs and outputs of any primitive site it encloses,
# and returns a dictionary containing that data to the user.
class trace(Messenger):
    def __enter__(self):
        super().__enter__()
        self.trace = OrderedDict()
        return self.trace

    # trace illustrates why we need postprocess_message in addition to process_message:
    # We only want to record a value after all other effects have been applied
    def postprocess_message(self, msg):
        assert msg["type"] != "sample" or msg["name"] not in self.trace, \
            "sample sites must have unique names"
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


# A second example of an effect handler for setting the value at a sample site.
# This illustrates why effect handlers are a useful PPL implementation technique:
# We can compose trace and replay to replace values but preserve distributions,
# allowing us to compute the joint probability density of samples under a model.
# See the definition of elbo(...) below for an example of this pattern.
class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super().__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


# block allows the selective application of effect handlers to different parts of a model.
# Sites hidden by block will only have the handlers below block on the PYRO_STACK applied,
# allowing inference or other effectful computations to be nested inside models.
class block(Messenger):
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super().__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True


# seed is used to fix the RNG state when calling a model.
class seed(Messenger):
    def __init__(self, fn=None, rng_seed=None):
        self.rng_seed = rng_seed
        super().__init__(fn)

    def __enter__(self):
        self.old_state = {'torch': torch.get_rng_state(), 'random': random.getstate()}
        torch.manual_seed(self.rng_seed)
        random.seed(self.rng_seed)
        try:
            import numpy as np
            np.random.seed(self.rng_seed)
            self.old_state['numpy'] = np.random.get_state()
        except ImportError:
            pass

    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.old_state['torch'])
        random.setstate(self.old_state['random'])
        if 'numpy' in self.old_state:
            import numpy as np
            np.random.set_state(self.old_state['numpy'])


# This limited implementation of PlateMessenger only implements broadcasting.
class PlateMessenger(Messenger):
    def __init__(self, fn, size, dim):
        assert dim < 0
        self.size = size
        self.dim = dim
        super().__init__(fn)

    def process_message(self, msg):
        if msg["type"] == "sample":
            batch_shape = msg["fn"].batch_shape
            if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
                batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
                batch_shape[self.dim] = self.size
                msg["fn"] = msg["fn"].expand(torch.Size(batch_shape))

    def __iter__(self):
        return range(self.size)


# apply_stack is called by pyro.sample and pyro.param.
# It is responsible for applying each Messenger to each effectful operation.
def apply_stack(msg):
    for pointer, handler in enumerate(reversed(PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


# sample is an effectful version of Distribution.sample(...)
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def sample(name, fn, *args, **kwargs):
    obs = kwargs.pop('obs', None)

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        return fn(*args, **kwargs)

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": args,
        "kwargs": kwargs,
        "value": obs,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


# param is an effectful version of PARAM_STORE.setdefault that also handles constraints.
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def param(name, init_value=None, constraint=torch.distributions.constraints.real, event_dim=None):
    if event_dim is not None:
        raise NotImplementedError("minipyro.plate does not support the event_dim arg")

    def fn(init_value, constraint):
        if name in PARAM_STORE:
            unconstrained_value, constraint = PARAM_STORE[name]
        else:
            # Initialize with a constrained value.
            assert init_value is not None
            with torch.no_grad():
                constrained_value = init_value.detach()
                unconstrained_value = torch.distributions.transform_to(constraint).inv(constrained_value)
            unconstrained_value.requires_grad_()
            PARAM_STORE[name] = unconstrained_value, constraint

        # Transform from unconstrained space to constrained space.
        constrained_value = torch.distributions.transform_to(constraint)(unconstrained_value)
        constrained_value.unconstrained = weakref.ref(unconstrained_value)
        return constrained_value

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        return fn(init_value, constraint)

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value, constraint),
        "value": None,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


# boilerplate to match the syntax of actual pyro.plate:
def plate(name, size, dim=None):
    if dim is None:
        raise NotImplementedError("minipyro.plate requires a dim arg")
    return PlateMessenger(fn=None, size=size, dim=dim)


# This is a thin wrapper around the `torch.optim.Adam` class that
# dynamically generates optimizers for dynamically generated parameters.
# See http://docs.pyro.ai/en/stable/optimization.html
class Adam:
    def __init__(self, optim_args):
        self.optim_args = optim_args
        # Each parameter will get its own optimizer, which we keep track
        # of using this dictionary keyed on parameters.
        self.optim_objs = {}

    def __call__(self, params):
        for param in params:
            # If we've seen this parameter before, use the previously
            # constructed optimizer.
            if param in self.optim_objs:
                optim = self.optim_objs[param]
            # If we've never seen this parameter before, construct
            # an Adam optimizer and keep track of it.
            else:
                optim = torch.optim.Adam([param], **self.optim_args)
                self.optim_objs[param] = optim
            # Take a gradient step for the parameter param.
            optim.step()


# This is a unified interface for stochastic variational inference in Pyro.
# The actual construction of the loss is taken care of by `loss`.
# See http://docs.pyro.ai/en/stable/inference_algos.html
class SVI:
    def __init__(self, model, guide, optim, loss):
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    # This method handles running the model and guide, constructing the loss
    # function, and taking a gradient step.
    def step(self, *args, **kwargs):
        # This wraps both the call to `model` and `guide` in a `trace` so that
        # we can record all the parameters that are encountered. Note that
        # further tracing occurs inside of `loss`.
        with trace() as param_capture:
            # We use block here to allow tracing to record parameters only.
            with block(hide_fn=lambda msg: msg["type"] == "sample"):
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        # Differentiate the loss.
        loss.backward()
        # Grab all the parameters from the trace.
        params = [site["value"].unconstrained()
                  for site in param_capture.values()]
        # Take a step w.r.t. each parameter in params.
        self.optim(params)
        # Zero out the gradients so that they don't accumulate.
        for p in params:
            p.grad = torch.zeros_like(p)
        return loss.item()


# This is a basic implementation of the Evidence Lower Bound, which is the
# fundamental objective in Variational Inference.
# See http://pyro.ai/examples/svi_part_i.html for details.
# This implementation has various limitations (for example it only supports
# random variables with reparameterized samplers), but all the ELBO
# implementations in Pyro share the same basic logic.
def elbo(model, guide, *args, **kwargs):
    # Run the guide with the arguments passed to SVI.step() and trace the execution,
    # i.e. record all the calls to Pyro primitives like sample() and param().
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    # Now run the model with the same arguments and trace the execution. Because
    # model is being run with replay, whenever we encounter a sample site in the
    # model, instead of sampling from the corresponding distribution in the model,
    # we instead reuse the corresponding sample from the guide. In probabilistic
    # terms, this means our loss is constructed as an expectation w.r.t. the joint
    # distribution defined by the guide.
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    # We will accumulate the various terms of the ELBO in `elbo`.
    elbo = 0.
    # Loop over all the sample sites in the model and add the corresponding
    # log p(z) term to the ELBO. Note that this will also include any observed
    # data, i.e. sample sites with the keyword `obs=...`.
    for site in model_trace.values():
        if site["type"] == "sample":
            elbo = elbo + site["fn"].log_prob(site["value"]).sum()
    # Loop over all the sample sites in the guide and add the corresponding
    # -log q(z) term to the ELBO.
    for site in guide_trace.values():
        if site["type"] == "sample":
            elbo = elbo - site["fn"].log_prob(site["value"]).sum()
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo


# This is a wrapper for compatibility with full Pyro.
def Trace_ELBO(**kwargs):
    return elbo


# This is a Jit wrapper around elbo() that (1) delays tracing until the first
# invocation, and (2) registers pyro.param() statements with torch.jit.trace.
# This version does not support variable number of args or non-tensor kwargs.
class JitTrace_ELBO:
    def __init__(self, **kwargs):
        self.ignore_jit_warnings = kwargs.pop("ignore_jit_warnings", False)
        self._compiled = None
        self._param_trace = None

    def __call__(self, model, guide, *args):
        # On first call, initialize params and save their names.
        if self._param_trace is None:
            with block(), trace() as tr, block(hide_fn=lambda m: m["type"] != "param"):
                elbo(model, guide, *args)
            self._param_trace = tr

        # Augment args with reads from the global param store.
        unconstrained_params = tuple(param(name).unconstrained()
                                     for name in self._param_trace)
        params_and_args = unconstrained_params + args

        # On first call, create a compiled elbo.
        if self._compiled is None:

            def compiled(*params_and_args):
                unconstrained_params = params_and_args[:len(self._param_trace)]
                args = params_and_args[len(self._param_trace):]
                for name, unconstrained_param in zip(self._param_trace, unconstrained_params):
                    constrained_param = param(name)  # assume param has been initialized
                    assert constrained_param.unconstrained() is unconstrained_param
                    self._param_trace[name]["value"] = constrained_param
                return replay(elbo, guide_trace=self._param_trace)(model, guide, *args)

            with validation_enabled(False), warnings.catch_warnings():
                if self.ignore_jit_warnings:
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                self._compiled = torch.jit.trace(compiled, params_and_args, check_trace=False)

        return self._compiled(*params_and_args)
Example use of mini-Pyro
View examples/minipyro.py on github

# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the functionality of `pyro.contrib.minipyro`,
which is a minimal implementation of the Pyro Probabilistic Programming
Language that was created for didactic purposes.
"""

import argparse

import torch

from pyro.generic import distributions as dist
# We use the pyro.generic interface to support dynamic choice of backend.
from pyro.generic import infer, ops, optim, pyro, pyro_backend


def main(args):
    # Define a basic model with a single Normal latent random variable `loc`
    # and a batch of Normally distributed observations.
    def model(data):
        loc = pyro.sample("loc", dist.Normal(0., 1.))
        with pyro.plate("data", len(data), dim=-1):
            pyro.sample("obs", dist.Normal(loc, 1.), obs=data)

    # Define a guide (i.e. variational distribution) with a Normal
    # distribution over the latent random variable `loc`.
    def guide(data):
        guide_loc = pyro.param("guide_loc", torch.tensor(0.))
        guide_scale = ops.exp(pyro.param("guide_scale_log", torch.tensor(0.)))
        pyro.sample("loc", dist.Normal(guide_loc, guide_scale))

    # Generate some data.
    torch.manual_seed(0)
    data = torch.randn(100) + 3.0

    # Because the API in minipyro matches that of Pyro proper,
    # training code works with generic Pyro implementations.
    with pyro_backend(args.backend):
        # Construct an SVI object so we can do variational inference on our
        # model/guide pair.
        Elbo = infer.JitTrace_ELBO if args.jit else infer.Trace_ELBO
        elbo = Elbo()
        adam = optim.Adam({"lr": args.learning_rate})
        svi = infer.SVI(model, guide, adam, elbo)

        # Basic training loop
        pyro.get_param_store().clear()
        for step in range(args.num_steps):
            loss = svi.step(data)
            if step % 100 == 0:
                print("step {} loss = {}".format(step, loss))

        # Report the final values of the variational parameters
        # in the guide after training.
        for name in pyro.get_param_store():
            value = pyro.param(name)
            print("{} = {}".format(name, value.detach().cpu().numpy()))

        # For this simple (conjugate) model we know the exact posterior. In
        # particular we know that the variational distribution should be
        # centered near 3.0. So let's check this explicitly.
        assert (pyro.param("guide_loc") - 3.0).abs() < 0.1


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-b", "--backend", default="minipyro")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    main(args)