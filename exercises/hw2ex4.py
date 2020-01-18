#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# ME210B - Homework 2, Exercise 4
#

import numpy as np
import tikzplotlib

import odesolvers

def hw2ex4ode(t, x):
	"""Function containing the ODE.

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
	xprime[1] = -100*(x[1] - np.sin(t)) + np.cos(t);

	return xprime;

if __name__ == '__main__':
	iv = np.array([1.0,2.0]);
	t0 : np.float = 0.0;
	tn : np.float = 1.0;
	h1 : np.float = 0.01;
	h2 : np.float = 0.05;

	# solving the ODE with h = 0.01
	y1 = odesolvers.ExplicitEulerSolver(hw2ex4ode, iv, t0, tn, h1);

	# plotting the results
	odesolvers.plotODEsol(y1[:,0], t0, h1, 'y1(t)');
	tikzplotlib.save(f'y1-{h1}.tex');
	odesolvers.plotODEsol(y1[:,1], t0, h1, 'y2(t)');
	tikzplotlib.save(f'y2-{h1}.tex');

	# solving the ODE with h = 0.05
	y2 = odesolvers.ExplicitEulerSolver(hw2ex4ode, iv, t0, tn, h2);

	# plotting the results
	odesolvers.plotODEsol(y2[:,0], t0, h2, 'y1(t)');
	tikzplotlib.save(f'y1-{h2}.tex');
	odesolvers.plotODEsol(y2[:,1], t0, h2, 'y2(t)');
	tikzplotlib.save(f'y2-{h2}.tex');
