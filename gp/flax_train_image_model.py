from dataset import *
from jax_models import *
from gpax import *
import optax
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze
import flax
import jax.numpy.linalg as linalg
import jax.numpy as np
from jax import random, device_put, vmap, vjp, jit
import jax
from absl import app, flags, logging
from ml_collections import config_flags
from clu import metric_writers, periodic_actions
from functools import partial
from typing import (Any, Callable, Sequence, Optional,
                    Tuple, Union, List, Iterable)
import os
import time
os.environ['KMP_WARNINGS'] = '0'
os.environ['LD_LIBRARY_PATH'] = '${LD_LIBRARY_PATH}:/usr/local/cuda/lib64'


FLAGS = flags.FLAGS
flags.DEFINE_string('work_dir', None, 'Directory to store data')
config_flags.DEFINE_config_file('config')


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
    metrics = {'loss': loss,
               'accuracy': metrics['accuracy']}
    return state, metrics


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


def create_train_state(rng, config, model):
    model_key, state_key = random.split(rng)

    variables = model.init(
        model_key, np.ones((1,)+config.image_shape))

    tx = optax.sgd(config.learning_rate,
                   nesterov=True)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'] if
        'batch_stats' in variables else {},
        rng=state_key)

    return state


def train_and_evaluate(config, work_dir):

    key = random.PRNGKey(0)
    key, state_key, data_key = random.split(key, 3)

    # logging

    writer = metric_writers.create_default_writer(
        logdir=config.metrics_dir)

    # dataset

    Y_subset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train, y_train, X_test, y_test = get_dataset(
        config.dataset, Y_subset=Y_subset)

    X_train, y_train, X_test, y_test = map(
        lambda x: jax_to_gpu(np.asarray(x)),
        [X_train, y_train, X_test, y_test])

    data_train = (X_train, y_train)
    data_test = (X_test, y_test)
    logging.info(f'shape (X_train, y_train): '
                 f'{X_train.shape}, {y_train.shape}')

    train_n_batches, train_batches = get_data_stream(
        data_key, config.batch_size, data_train)

    ## model & state

    model_def = get_jax_model_def(config.model)
    model = model_def(num_classes=config.num_classes)

    state = create_train_state(state_key, config, model)
    step_offset, state = checkpoint_restore(work_dir, state)
    step_offset = 0 if step_offset is None \
        else step_offset+1

    logging.info(f'Start at step_offset={step_offset}')

    # training/evaluation loop

    hooks = [
        periodic_actions.ReportProgress(
            num_train_steps=config.n_epochs*train_n_batches,
            every_steps=train_n_batches-1, writer=writer),
        periodic_actions.Profile(logdir=config.metrics_dir,
                                 num_profile_steps=5)
    ]
    train_metrics = []

    for epoch in range(step_offset, config.n_epochs):
        for it in range(train_n_batches):
            batch = next(train_batches)
            step = epoch*train_n_batches+it
            for h in hooks:
                h(step)

            state, metrics = train_step(state, batch)
            train_metrics.append(metrics)

            if (step+1) % (train_n_batches//10) == 0:
                train_metrics = jax.tree_map(lambda x: x.mean(),
                                             stack_forest(train_metrics))
                train_metrics['epoch'] = step/train_n_batches
                writer.write_scalars(step+1, {f'train/{k}': v
                                              for k, v in train_metrics.items()})
                train_metrics = []

        checkpoint_save(work_dir, state, step=epoch)
        test_metrics = eval_model(state, data_test)
        test_metrics['epoch'] = epoch
        writer.write_scalars(epoch+1, {f'eval/{k}': v
                                       for k, v in test_metrics.items()})
        writer.flush()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.set_verbosity(logging.INFO)

    config = FLAGS.config
    work_dir = FLAGS.work_dir if (FLAGS.work_dir is not None) \
        else f'./results/{config.dataset}_{config.model}'

    with config.unlocked():
        config.work_dir = work_dir
        config.metrics_dir = os.path.join(config.work_dir, 'metrics')
        config.ckpt_dir = os.path.join(config.work_dir, 'checkpoints')

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    logging.info('JAX process: %d / %d',
                 jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    train_and_evaluate(config, work_dir)


if __name__ == '__main__':
    app.run(main)
