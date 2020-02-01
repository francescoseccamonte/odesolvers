#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# File implementing the theta method for ODEs numerical solution.
#

import numpy as np
from nptyping import Array

from ._expliciteuler import _ExplicitEuler_step
from ._backwardeuler import _BackwardEuler_step

def ThetaMethod(f, iv : Array[float], t0 : float, tn : float, h : float, theta : float, df = None, TOL : float = 1.0e-5, NEWTITER : int = 10) -> Array[float]:
	"""Function implementing the Theta method for ODEs numerical solution.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions
	Theta method implements the numerical scheme: x_{n+1} = x_n + theta h f(t_n,x_n) + (1 - theta)h f(t_{n+1},x_{n+1})

	Theta = 1 is equivalent to the Explicit (Forward) Euler method.
	Theta = 0 is equivalent to the Implicit (Backward) Euler method.

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param iv: vector of initial values
		:param t0: initial time
		:param tn: final time
		:param h: step size
		:param theta: value between 0 and 1
		:param df: Jacobian of f
		:param TOL: Numerical tolerance for convergence
		:param NEWTITER: Maximum number of Newton iterations to be performed
		:type f: Callable
		:type iv: np.array[float]
		:type t0: np.float
		:type tn: np.float
		:type h: np.float
		:type theta: np.float
		:type df: Callable
		:type TOL: np.float
		:type NEWTITER: (unsigned) int
		:return: Vector x containing solution of component j at time i (x[i,j])
		:rtype: np.array[float,float]

	"""

	if h <= 0.0:
		raise ValueError('The stepsize h must be positive')

	if (tn - t0) <= 0.0:
		raise ValueError('The final time must be greater than the initial time')

	if not 0 <= theta <= 1:
		raise ValueError('Theta has to be between 0 and 1')

	if TOL <= 0.0:
		raise ValueError('The numerical tolerance must be positive')

	if (theta != 1) and (NEWTITER < 0.0):
		raise ValueError('The maximum number of Newton Iteration steps must be positive')

	if (theta != 1) and df is None:
		raise NotImplementedError('Automatic differentiation not implemented yet. Please provide the Jacobian of f')

	N : np.uint = np.uint(np.ceil((tn - t0)/h));	# number of steps

	x = np.empty((np.uint(N+1),iv.size), float);	# preallocating the array (+1 for including initial condition)
	x[0,:] = iv;

	if (theta == 1):
		for i in range(N):
			x[i+1,:] = _ExplicitEuler_step(f,x[i,:],(t0+h*i),h);
	elif (theta == 0):
		for i in range(N):
			x[i+1,:] = _BackwardEuler_step(f,df,x[i,:],(t0+h*i),h,TOL,NEWTITER);
	else:
		for i in range(N):
			x[i+1,:] = theta*_ExplicitEuler_step(f,x[i,:],(t0+h*i),h) + (1 - theta)*_BackwardEuler_step(f,df,x[i,:],(t0+h*i),h,TOL,NEWTITER);

	return x;
