#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# ME210B - Homework 5
#

import numpy as np
from scipy.linalg import toeplitz
import tikzplotlib

import odesolvers

from hw4ex1 import hw4ex1ode2


def hw5ode1(t, x):
	"""Function containing the first ODE ("easy problem").

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Derivative of state at current time
		:rtype: np.array[float]

	"""
	xprime = np.empty([2], float);

	xprime[0] = -x[0];
	xprime[1] = -10*x[1] +2*t*(5*t + 1);

	return xprime;

def hw5ode2(t, x):
	"""Function containing the second ODE.

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Derivative of state at current time
		:rtype: np.array[float]

	"""

	return hw4ex1ode2(t,x);

def hw5ode3(t, x):
	"""Function containing the third ODE
	(Van der Pol’s equation with \\eta = 2).

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Derivative of state at current time
		:rtype: np.array[float]

	"""
	xprime = np.empty([2], float);

	xprime[0] = x[1];
	xprime[1] = 2*((1 - x[0]*x[0])*x[1] - x[0]);

	return xprime;

def hw5pde(t, x):
	"""Function containing the PDE to be solved
	with method of lines (gridding here implemented)
	and Backward Euler difference.

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Derivative of state at current time
		:rtype: np.array[float]

	"""
	n : np.int = 100;		# number of points in the grid
	r = np.zeros(n);
	c = np.zeros(n);
	r[0] = c[0] = -n;
	c[1] = n;
	A = toeplitz(c, r);
	ub = np.zeros(n); ub[0] = n;	# constant offset given by boundary condition

	return (np.dot(A,x) + ub);


if __name__ == '__main__':

	ETOL = np.array([1e-3, 1e-6]);

	for tol in np.nditer(ETOL):
		# Problem 1
		iv = np.array([1.0,2.0]);
		t0 : np.float = 0.0;
		tn : np.float = 1.0;

		# Fixed stepsize
		h : np.float = 0.01;
		y, hi = odesolvers.AB_AM_PECE2(hw5ode1, iv, t0, tn, None, ETOL=tol);
		odesolvers.plotODEsol(y[:,0], t0, h, 'y1(t)');
		tikzplotlib.save(f'problem1-y1-tol-{tol}-step-{h}.tex');
		odesolvers.plotODEsol(y[:,1], t0, h, 'y2(t)');
		tikzplotlib.save(f'problem1-y2-tol-{tol}-step-{h}.tex');

		# Automatic stepsize selection
		y, hi = odesolvers.AB_AM_PECE2(hw5ode1, iv, t0, tn, None, ETOL=tol);
		odesolvers.plotODEsolVar(y[:,0], t0, hi, 'y1(t)');
		tikzplotlib.save(f'problem1-y1-tol-{tol}-variable-step.tex');
		odesolvers.plotODEsolVar(y[:,1], t0, hi, 'y2(t)');
		tikzplotlib.save(f'problem1-y2-tol-{tol}-variable-step.tex');
		odesolvers.plotODEsolVar(hi[1:], t0+hi[1], hi[1:], 'h');
		tikzplotlib.save(f'problem1-step-tol-{tol}-variable-step.tex');


		# Problem 2
		iv = np.array([10.0,20.0]);
		tn : np.float = 100.0;

		y, hi = odesolvers.AB_AM_PECE2(hw5ode2, iv, t0, tn, None, ETOL=tol);
		odesolvers.plotODEsolVar(y[:,0], t0, hi, 'y1(t)');
		tikzplotlib.save(f'problem2-y1-tol-{tol}-variable-step.tex');
		odesolvers.plotODEsolVar(y[:,1], t0, hi, 'y2(t)');
		tikzplotlib.save(f'problem2-y2-tol-{tol}-variable-step.tex');
		odesolvers.ODEphaseplot(y[:,0], y[:,1], t0, None, 'y1(t)', 'y2(t)');
		tikzplotlib.save(f'problem2-y1y2-tol-{tol}-variable-step.tex');
		odesolvers.plotODEsolVar(hi[1:], t0+hi[1], hi[1:], 'h');
		tikzplotlib.save(f'problem2-step-tol-{tol}-variable-step.tex');

		# Problem 3
		iv = np.array([2.0,0.0]);
		tn : np.float = 11.0;

		y, hi = odesolvers.AB_AM_PECE2(hw5ode3, iv, t0, tn, None, ETOL=tol);
		odesolvers.plotODEsolVar(y[:,0], t0, hi, 'y1(t)');
		tikzplotlib.save(f'problem3-y1-tol-{tol}-variable-step.tex');
		odesolvers.plotODEsolVar(y[:,1], t0, hi, 'y2(t)');
		tikzplotlib.save(f'problem3-y2-tol-{tol}-variable-step.tex');
		odesolvers.plotODEsolVar(hi[1:], t0+hi[1], hi[1:], 'h');
		tikzplotlib.save(f'problem3-step-tol-{tol}-variable-step.tex');

		# Problem 4
		tn : np.float = 1.0;
		bins : np.int = 101;
		iv = np.linspace(0.0, 1.0, num=bins);
		iv = np.exp(-10*iv);

		yp, hi = odesolvers.AB_AM_PECE2(hw5pde, iv[1:], t0, tn, None, ETOL=tol);
		y = np.concatenate((iv[0]*np.ones((hi.size,1)),yp), axis=1);	# adding boundary value
		odesolvers.plotODEsolVar(hi[1:], t0+hi[1], hi[1:], 'h');
		tikzplotlib.save(f'problem4-step-tol-{tol}-variable-step.tex');

		T = np.array([0, 0.25, 0.5, 0.6, 0.8, 1.0]);

		for t in np.nditer(T):
			xp = odesolvers.AB_AM_PECE2_interpatT(hw5pde,t,(t0 + np.cumsum(hi)), yp);
			xp = np.ravel(xp);
			xT = np.concatenate(([1.0],xp));	# adding boundary value
			odesolvers.ODEphaseplot(np.linspace(0.0, 1.0, num=bins), xT, None, None, 'x', 'f(x)');
			tikzplotlib.save(f'problem4-step-tol-{tol}-t-{t}-variable-step.tex');

