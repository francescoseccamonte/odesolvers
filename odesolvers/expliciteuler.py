#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# File implementing the Explicit Euler method for ODEs numerical solution.
#

import numpy as np
from typing import Callable
from nptyping import Array

ode = Callable[[float, Array[float]], Array[float]]

def ExplicitEulerSolver(f : ode, iv : Array[float], t0 : float, tn : float, h : float) -> Array[float]:
	"""Function implementing the Explicit Euler method for ODEs numerical solution.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param iv: vector of initial values
		:param t0: initial time
		:param tn: final time
		:param h: step size
		:type f: ode = Callable[[float, Array[float]], Array[float]]
		:type iv: np.array[float]
		:type t0: np.float
		:type tn: np.float
		:type h: np.float
		:return: Vector x containing solution of component j at time i (x[i,j])
		:rtype: np.array[float,float]

	"""

	if h <= 0.0:
		raise ValueError('The stepsize h must be positive')

	if (tn - t0) <= 0.0:
		raise ValueError('The final time must be greater than the initial time')

	N : np.uint = np.uint(np.ceil((tn - t0)/h));	# number of steps

	x = np.empty((np.uint(N+1),iv.size), float);	# preallocating the array (+1 for including initial condition)
	x[0,:] = iv;

	for i in range(N):
		x[i+1,:] = x[i,:] + h*f((t0+h*i), x[i,:]);

	return x;
