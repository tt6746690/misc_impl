import os, time
os.environ['KMP_WARNINGS'] = '0'
os.environ['LD_LIBRARY_PATH'] = '${LD_LIBRARY_PATH}:/usr/local/cuda/lib64'

from typing import (Any, Callable, Sequence, Optional, Tuple, Union, List, Iterable)
from functools import partial
from collections import defaultdict
from ml_collections import config_flags

from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string('work_dir', None, 'Directory to store data')
config_flags.DEFINE_config_file('config')

import jax
from jax import random, device_put, vmap, vjp, jit
import jax.numpy as np
import jax.numpy.linalg as linalg

import flax
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

import optax

from gpax import *
from jax_models import *
from dataset import *


class TrainState(train_state.TrainState):
    """ Keeps track of parameters, optimizer state, rng, 
            mutable states in training 
    """
    # `TrainState` attributes: 
    # step: int
    # apply_fn: Callable = struct.field(pytree_node=False)
    # params: core.FrozenDict[str, Any]
    # tx: optax.GradientTransformation = struct.field(pytree_node=False)
    # opt_state: optax.OptState
    #
    batch_stats: flax.core.FrozenDict[str, Any]
    rng: np.ndarray
        
    # jitted model's apply_fn for evaluation, use
    #     `struct.field(pytree_node=False)` to avoid type checks
    apply_fn_eval_jitted: Callable = flax.struct.field(pytree_node=False,
                                                       default=None)
        
    @property
    def variables(self):
        return freeze({'params': self.params,
                       'batch_stats': self.batch_stats})

    @property
    def apply_fn_eval(self):
        """ Lazy jitting of `apply_fn` for model eval
        """
        if self.apply_fn_eval_jitted is None:
            import inspect
            apply_fn = self.apply_fn
            # sig = inspect.signature(apply_fn)
            # kwargs = {'train': False} \
            #     if 'train' in sig.parameters else {}
            kwargs = {'train': False, 'mutable': False}
            apply_fn = partial(apply_fn, **kwargs)
            apply_fn = jax.jit(apply_fn)
            # Use `object.__setattr__` to mutate 
            #     fields in @dataclass(frozen=True)
            object.__setattr__(
                self, 'apply_fn_eval_jitted', apply_fn)
        return self.apply_fn_eval_jitted



@jax.jit
def compute_metrics(logits, labels):
    label_onehot = jax.nn.one_hot(labels.squeeze(), 10)
    loss = np.mean(optax.softmax_cross_entropy(logits=logits,
                                               labels=label_onehot))
    pred = np.argmax(logits, -1).reshape(labels.shape)
    accuracy = np.mean(pred == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    return metrics


@jax.jit
def train_step(state, batch):
    state_rng, train_rng = random.split(state.rng)
    X, y = batch
    def loss_fn(params):
        y_onehot = jax.nn.one_hot(y, 10).squeeze()
        logits, mutable_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            X, mutable=['batch_stats'])
        # mutable_state: {'batch_stats': {...}}
        loss = np.mean(optax.softmax_cross_entropy(logits=logits,
                                                   labels=y_onehot))
        aux = (logits, mutable_state)
        return loss, aux
    grad_fn = jax.value_and_grad(loss_fn,
                                 has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    logits, mutable_state = aux
    # `state.apply_gradients(grads)` does 3 things 
    #     - compute updates given gradient transformation `state.tx` & `grads`
    #     - apply the updates using `optax.apply_updates`
    #     - create new `state` with mutated `step`, `params`, `opt_state`, etc.
    state = state.apply_gradients(
        grads=grads,
        batch_stats=mutable_state['batch_stats'],
        rng=state_rng)
    metrics = compute_metrics(logits, y)
    log = {'loss': loss,
           'accuracy': metrics['accuracy']}
    return state, log



def eval_model(state, data_test):
    test_n_batches, test_batches = get_data_stream(
        random.PRNGKey(0), 100, data_test)

    logits = []
    labels = []
    for _ in range(test_n_batches):
        batch = next(test_batches)
        X, y = batch
        logit = state.apply_fn_eval(state.variables,
                                    X)
        labels.append(y.reshape(-1, 1))
        logits.append(logit)

    logits = np.vstack(logits)
    labels = np.vstack(labels)
    metrics = compute_metrics(logits, labels)
    metrics = jax.tree_map(lambda x: x.item(),
                           metrics)
    return metrics



def train_and_evaluate(config, work_dir):

    key = random.PRNGKey(0)
    model_key, state_key, data_key = random.split(key, 3)
    
    ## dataset
    
    Y_subset = [0,1,2,3,4,5,6,7,8,9]
    X_train, y_train, X_test, y_test = get_dataset(
        config.dataset, Y_subset=Y_subset)

    X_train, y_train, X_test, y_test = map(
        lambda x: jax_to_gpu(np.asarray(x)),
        [X_train, y_train, X_test, y_test])

    data_train = (X_train, y_train)
    data_test   = (X_test, y_test)
    print('shape (X_train, y_train): ', X_train.shape, y_train.shape)


    ## model

    model_def = get_jax_model_def(config.model)
    model = model_def(num_classes=config.num_classes)

    variables = model.init(
        model_key, np.ones((1,)+config.image_shape))
    
    ## optimizer and state

    tx = optax.sgd(config.learning_rate,
                   nesterov=True)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'] if \
            'batch_stats' in variables else {},
        rng=state_key)

    train_n_batches, train_batches = get_data_stream(
        data_key, config.batch_size, data_train)
    
    ## training/evaluation loop

    for epoch in range(config.n_epochs):
        logs = defaultdict(list)
        for it in range(train_n_batches):
            step = epoch*train_n_batches+it
            batch = next(train_batches)
            state, log = train_step(state, batch)
            variables = state.variables

            for k, v in log.items():
                logs[k].append(v)
            if step%(train_n_batches//10)==0:
                avg_metrics = {k: np.mean(np.array(v))
                               for k, v in logs.items()}
                print(f'[{epoch:3}|{100*it/train_n_batches:5.2f}%]\t'
                      f'Loss={avg_metrics["loss"]:.3f}\t'
                      f'accuracy={avg_metrics["accuracy"]:.3f}\t')


        checkpoint_save(work_dir, state, step=epoch)
        metrics = eval_model(state, data_test)
        print(f'[{epoch:3}] test \t'
              f'Loss={metrics["loss"]:.3f}\t'
              f'accuracy={metrics["accuracy"]:.3f}\t')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    config = FLAGS.config
    work_dir = FLAGS.work_dir if ( FLAGS.work_dir is not None ) \
        else  f'./results/{config.dataset}_{config.model}'

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    train_and_evaluate(config, work_dir)


if __name__ == '__main__':
    app.run(main)


