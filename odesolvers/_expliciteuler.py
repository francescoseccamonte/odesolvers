#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Internal implementation of the Explicit Euler method for ODEs numerical solution.
#

def _ExplicitEuler_step(f, xi, ti, h):
	"""Internal function implementing one step of the Explicit Euler method.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param xi: current time
		:param h: step size
		:type f: Callable
		:type xi: np.array[float]
		:type ti: np.float
		:type h: np.float
		:return: Vector x containing solution of component j at current time ti (x[j])
		:rtype: np.array[float]

	"""

	xnext = xi + h*f(ti, xi);

	return xnext;
