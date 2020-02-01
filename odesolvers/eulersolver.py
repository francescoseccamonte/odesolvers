#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# File implementing the Explicit and Implicit Euler methods for ODEs numerical solution.
#

from nptyping import Array

from .thetamethod import ThetaMethod

def ExplicitEulerSolver(f, iv : Array[float], t0 : float, tn : float, h : float) -> Array[float]:
	"""Function implementing the Explicit Euler method for ODEs numerical solution.
	It leverages the ThetaMethod function.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param iv: vector of initial values
		:param t0: initial time
		:param tn: final time
		:param h: step size
		:type f: Callable
		:type iv: np.array[float]
		:type t0: np.float
		:type tn: np.float
		:type h: np.float
		:return: Vector x containing solution of component j at time i (x[i,j])
		:rtype: np.array[float,float]

	"""

	return ThetaMethod(f, iv, t0, tn, h, 1);


def ImplicitEulerSolver(f, iv : Array[float], t0 : float, tn : float, h : float, df = None, TOL : float = 1.0e-5, NEWTITER : int = 10) -> Array[float]:
	"""Function implementing the Explicit Euler method for ODEs numerical solution.
	It leverages the ThetaMethod function.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param iv: vector of initial values
		:param t0: initial time
		:param tn: final time
		:param h: step size
		:param df: Jacobian of f
		:param TOL: Numerical tolerance for convergence
		:param NEWTITER: Maximum number of Newton iterations to be performed
		:type f: Callable
		:type iv: np.array[float]
		:type t0: np.float
		:type tn: np.float
		:type h: np.float
		:type df: Callable
		:type TOL: np.float
		:type NEWTITER: (unsigned) int
		:return: Vector x containing solution of component j at time i (x[i,j])
		:rtype: np.array[float,float]

	"""

	return ThetaMethod(f, iv, t0, tn, h, 0, df, TOL, NEWTITER);

