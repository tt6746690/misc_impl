def jpt_in_notebook():
    try:
        from IPython import get_ipython
        import os

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except:
        return False
    else:  # pragma: no cover
        return True


def jpt_autoreload():
    if jpt_in_notebook():
        from IPython.core.getipython import get_ipython
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')


def jpt_full_width():
    if jpt_in_notebook():
        from IPython.core.display import display, HTML
        display(HTML("<style>.container { width:100% !important; }</style>"))


def jpt_suppress_warnings():
    import warnings
    from pandas.core.common import SettingWithCopyWarning
    if jpt_in_notebook():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", SettingWithCopyWarning)
    else:
        warnings.filterwarnings("ignore")


def jpt_convert_to_py(ipynb_path):
    if jpt_in_notebook():
        import shlex
        import subprocess
        cmd = f"jupyter nbconvert --to script {ipynb_path}"
        try:
            retcode = subprocess.call(shlex.split(cmd))
        except OSError as err:
            print(f'Execution failed: [{cmd}]')


def jpt_tsb_stop(port):
    import os
    # Use SIGTERM/15 instead of SIGKILL/9 so `tensorboard` does proper clean-ups
    os.system(f'kill -15 $(lsof -t -i:{port})')


def jpt_tsb_start(logdir, port):
    """Does same thing as `tensorboard` extension as Jupyter magics
            %tensorboard --logdir=$logdir --port=$port
    """
    from tensorboard import notebook
    notebook.start(f'--logdir="{logdir}" --port={port}')


def jpt_tsb_restart(logdir, port):
    if jpt_in_notebook():
        jpt_tsb_stop(port)
        jpt_tsb_start(logdir, port)


def jpt_check_memusage(globals_, dir_, getsize_opt='pympler', in_gb=True):
    """ Check memory usage of global variables 
        utils.jpt_check_memusage(globals(), dir())
    """
    import sys
    from pympler import asizeof

    if getsize_opt == 'pympler':
        getsize = asizeof.asizeof
    else:
        getsize = sys.getsizeof

    if jpt_in_notebook():
        # These are the usual ipython objects
        ipython_vars = ['In', 'Out', 'exit',
                        'quit', 'get_ipython', 'ipython_vars']
    else:
        ipython_vars = []

    l = []
    b = []
    for x in dir_:
        if x.startswith('_') or x in sys.modules or x in ipython_vars:
            continue
        try:
            size = getsize(globals_.get(x))
            if in_gb:
                size = size/1024**3
            l.append((x, size))
        except:
            b.append(x)
    l = sorted(l, key=lambda x: x[1], reverse=True)
    return l, b