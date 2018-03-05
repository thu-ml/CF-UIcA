import numpy as np
import time
import os
import errno
import logging
import coloredlogs
import theano
import theano.tensor as T
from collections import OrderedDict

def adam(grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    auxiliary_sym_vars = []
    all_grads = grads
    t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))
    updates = OrderedDict()
    auxiliary_sym_vars.append(t_prev)

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        auxiliary_sym_vars.append(m_prev)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        auxiliary_sym_vars.append(v_prev)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates, auxiliary_sym_vars

def get_all_shvar_values(auxiliary_sym_vars):
    auxiliary_sym_vars_values = [sym_var.get_value() for sym_var in auxiliary_sym_vars]
    return auxiliary_sym_vars_values

def set_all_shvar_values(auxiliary_sym_vars, auxiliary_sym_vars_values):
    for sym_var, value in zip(auxiliary_sym_vars, auxiliary_sym_vars_values):
        sym_var.set_value(value)

def get_dims(dims_str):
    if not dims_str:
        return []
    return [int(s) for s in dims_str.split(',')]

# parameters tagging helper function
def param_tag(value):
    if value == 0.0:
        return "00"
    exp = np.floor(np.log10(value))
    leading = ("%e"%value)[0]
    return "%s%d" % (leading, exp)

def setup_logging(exp_name, loglv):
    assert exp_name

    #FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = '%H:%M:%S'
    LEVEL_STYLES = dict(
        debug=dict(color='blue'),
        info=dict(color='green'),
        verbose=dict(),
        warning=dict(color='yellow'),
        error=dict(color='red'),
        critical=dict(color='magenta'))
    coloredlogs.install(level=loglv, fmt=FORMAT, datefmt=DATEFMT, level_styles=LEVEL_STYLES)

    # Determine suffix
    suffix = time.strftime("%Y-%m-%d--%H-%M")

    suffix_counter = 0
    dirname = "output/%s.%s" % (exp_name, suffix)
    while True:
        try:
            os.makedirs(dirname)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise e
            suffix_counter += 1
            dirname = "output/%s.%s+%d" % (exp_name, suffix, suffix_counter)
        else:
            break

    formatter = logging.Formatter(FORMAT, DATEFMT)
    logger_fname = os.path.join(dirname, "logfile.txt")
    fh = logging.FileHandler(logger_fname)
    #fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(fh)

    return dirname
