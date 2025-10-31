"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from pycparser.ply.ctokens import t_SEMI
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

class ForwardEuler(scipy.integrate.OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, vectorized=False,  support_complex=False, **extraneous):

        # default step size h
        self.h = extraneous.get('h', (t_bound - t0) / 100)

        # update h if it is set
        if 'h' in extraneous:
            extraneous.pop('h')

        # initialize the upper class
        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound,
                                           vectorized, support_complex)

        self.direction = 1
        self.y_old = self.y
        self.t_old = self.t
        self.nfev = 0   # number of function evaluation
        self.njev = 0   # maintain at 0 since we do not use Jacobian
        self.nlu = 0    # maintain at 0 since we do not use Jacobian


    def _step_impl(self):
        if self.status == 'finished':
            return True, None

        # grab current state
        current_t = self.t
        current_y = self.y
        current_h = self.h

        # propose next time using the current step size
        new_t = current_t + self.direction * current_h

        # if that would overshoot t_bound, clip the step
        if self.direction * (new_t - self.t_bound) > 0:
            new_t = self.t_bound
            current_h = new_t - current_t  # shorten last step size

        # evaluate f(t, y)
        current_f = self.fun(current_t, current_y)
        self.nfev += 1

        # forward euler update
        new_y = current_y + current_h * current_f

        # store old state for dense output
        self.t_old = current_t
        self.y_old = current_y

        # commit new state
        self.t = new_t
        self.y = new_y
        self.h = current_h

        # mark finished if we hit / passed bound
        if self.direction * (self.t - self.t_bound) >= 0:
            self.status = 'finished'

        return True, None

    def _dense_output_impl(self):
        return ForwardEulerOutput(self.t_old, self.t, self.y_old, self.y)


class ForwardEulerOutput(DenseOutput):
    def __init__(self, t_old, t_new, y_old, y_new):
        super(ForwardEulerOutput, self).__init__(t_old, t_new)
        # Initialization
        self.t_old = t_old
        self.t_new = t_new
        self.y_old = np.asarray(y_old)
        self.y_new = np.asarray(y_new)

    def _call_impl(self, t):

        t = np.asarray(t)

        # linear ratio s = (t - t_old)/(t_new - t_old)
        s = (t - self.t_old) / (self.t_new - self.t_old)

        # y_old, y_new: shape (n,)
        # we want output shape (n,) if t is scalar
        # or (n, k) if t is array of length k
        y_old = self.y_old.reshape(-1, 1)  # (n,1)
        y_new = self.y_new.reshape(-1, 1)  # (n,1)

        s_shape = s.shape
        s = s.reshape(1, -1)  # (1,k)

        y = y_old + s * (y_new - y_old)  # (n,k)

        if s_shape == ():  # scalar query
            return y[:, 0]
        else:
            return y

