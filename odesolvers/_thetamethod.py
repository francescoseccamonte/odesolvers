#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Internal implementation of the Theta method (including Backward Euler) for ODEs numerical solution.
#

import numpy as np

def _Theta_step(f, df, xi, ti, h, theta, TOL, MAXITER):
	"""Internal function implementing one step of the theta (including Backward Euler) method.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param df: Jacobian of f
		:param xi: initial condition at time ti
		:param ti: current time
		:param h: step size
		:param theta: value between 0 and 1
		:param TOL: Numerical tolerance for convergence
		:param MAXITER: Maximum number of Newton iterations to be performed
		:type f: Callable
		:type df: Callable
		:type xi: np.array[float]
		:type ti: np.float
		:type h: np.float
		:type theta: np.float
		:type TOL: np.float
		:type NEWTITER: (unsigned) int
		:return: Vector x containing solution of component j at next time ti+h (x[j])
		:rtype: np.array[float]

	"""

	# xinu represents the \nu step of the Newton iteration algorithm
	# xinu's first guess initialized as previous solution
	xinu = np.copy(xi);

	# Newton iteration
	for i in range(MAXITER):
		A = (np.identity(xi.size) - (1-theta)*h*df(ti+h,xinu));
		b = -(xinu - xi -theta*h*f(ti,xi) -(1-theta)*h*f(ti+h,xinu));

		# delta = xinu+1 - xinu
		# Solving the linear system of equations A delta = b, that is,
		# (I - (1-theta)*h*df)*delta = -(xinu - xi -theta*h*f(ti,xi) -(1-theta)*h*f(ti+h,xinu))
		delta = np.linalg.solve(A, b);

		xinu += delta;

		# check for convergence
		if np.linalg.norm(delta) <= TOL:
			return xinu;

	raise ArithmeticError('Newton iteration has not converged')
