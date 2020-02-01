#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Internal implementation of the Backward Euler method for ODEs numerical solution.
#

import numpy as np

def _BackwardEuler_step(f, df, xi, ti, h, TOL, MAXITER):
	"""Internal function implementing one step of the Backward Euler method.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param df: Jacobian of f
		:param xi: initial condition at time ti
		:param ti: current time
		:param h: step size
		:param TOL: Numerical tolerance for convergence
		:param MAXITER: Maximum number of Newton iterations to be performed
		:type f: Callable
		:type df: Callable
		:type xi: np.array[float]
		:type ti: np.float
		:type h: np.float
		:type TOL: np.float
		:type NEWTITER: (unsigned) int
		:return: Vector x containing solution of component j at current time ti (x[j])
		:rtype: np.array[float]

	"""

	# xinu represents the \nu step of the Newton iteration algorithm
	# xinu's first guess initialized as previous solution
	xinu = np.copy(xi);

	# Newton iteration # TODO CHECK
	for i in range(MAXITER):
		A = (np.identity(xi.size) - h*df(ti,xinu));
		b = -(xinu - xi - h*f(ti,xinu));

		# delta = xinu+1 - xinu
		# Solving the linear system of equations A delta = b, that is,
		# (I - h*df)*delta = -(xinu - xi - h*f(ti,xinu))
		delta = np.linalg.solve(A, b);

		xinu += delta;

		# check for convergence
		if np.linalg.norm(delta) <= TOL:
			return xinu;

	raise ArithmeticError('Newton iteration has not converged')
