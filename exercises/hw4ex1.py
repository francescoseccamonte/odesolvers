#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# ME210B - Homework 4, Exercise 1
#

import numpy as np
import tikzplotlib

import odesolvers

from hw2ex4 import hw2ex4ode

def hw4ex1ode1(t, x):
	"""Function containing the first ODE.

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Derivative of state at current time
		:rtype: np.array[float]

	"""

	return hw2ex4ode(t,x);


def hw4ex1Jacobian1(t, x):
	"""Function containing the Jacobian of the first ODE.

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Jacobian of the function hw4ex1ode1
		:rtype: np.array[float,float]

	"""

	df = np.empty([2,2], float);
	
	df[0,0] = -1;
	df[0,1] = 0;
	df[1,0] = 0;
	df[1,1] = -100;
	
	return df;

def hw4ex1ode2(t, x):
	"""Function containing the second ODE.

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Derivative of state at current time
		:rtype: np.array[float]

	"""
	xprime = np.empty([2], float);

	xprime[0] = 0.25*x[0] - 0.01*x[0]*x[1];
	xprime[1] = -x[1] + 0.01*x[0]*x[1];

	return xprime;

def hw4ex1Jacobian2(t, x):
	"""Function containing the Jacobian of the second ODE.

	- **parameters**, **types**, **return** and **return types**::
		:param t: current time
		:param x: state at current time
		:type t: np.float
		:type x: np.array[float]
		:return: Jacobian of the function hw4ex1ode2
		:rtype: np.array[float,float]

	"""

	df = np.empty([2,2], float);

	df[0,0] = 0.25 -0.01*x[1];
	df[0,1] = -0.01*x[0];
	df[1,0] = 0.01*x[1];
	df[1,1] = -1 + 0.01*x[0];
	
	return df;

if __name__ == '__main__':
	# Problem 1
	iv = np.array([1.0,2.0]);
	t0 : np.float = 0.0;
	tn : np.float = 1.0;

	theta = np.array([0, 0.5, 1]);
	h = np.array([0.01, 0.05]);

	for thetai in np.nditer(theta):
		for hi in np.nditer(h):
			y = odesolvers.ThetaMethod(hw4ex1ode1, iv, t0, tn, hi, thetai, hw4ex1Jacobian1);

			# plotting the results
			odesolvers.plotODEsol(y[:,0], t0, hi, 'y1(t)');
			# tikzplotlib.save(f'problem1-y1-step-{hi}-theta-{thetai}.tex');
			odesolvers.plotODEsol(y[:,1], t0, hi, 'y2(t)');
			# tikzplotlib.save(f'problem1-y2-step-{hi}-theta-{thetai}.tex');

	# Problem 2
	iv = np.array([10,10]);
	tn = 100;

	h = np.array([0.1, 0.001]);
	
	for thetai in np.nditer(theta):
		for hi in np.nditer(h):
			y = odesolvers.ThetaMethod(hw4ex1ode2, iv, t0, tn, hi, thetai, hw4ex1Jacobian2);

			# plotting the results
			odesolvers.plotODEsol(y[:,0], t0, hi, 'y1(t)');
			# tikzplotlib.save(f'problem2-y1-step-{hi}-theta-{thetai}.tex');
			odesolvers.plotODEsol(y[:,1], t0, hi, 'y2(t)');
			# tikzplotlib.save(f'problem2-y2-step-{hi}-theta-{thetai}.tex');

