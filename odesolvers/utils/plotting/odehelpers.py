#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# File containing helper functions for the ODE numerical solvers.
#

import numpy as np
import matplotlib.pyplot as plt
from nptyping import Array

def plotODEsol(x : Array[float], t0 : float, h : float, ylabel : str = 'x(t)') -> None:
	"""Function plotting a solution of an ODE.

	- **parameters**, **types**, **return** and **return types**::
		:param x: array containing ODE solution at time i x[i]
		:param t0: initial time
		:param h: step size
		:param ylabel: ylabel (default to x(t))
		:type x: np.array[float]
		:type t0: np.float
		:type h: np.float
		:type ylabel: string
		:return: None
		:rtype: None

	"""
	
	tn : np.float = (x.shape[0]-1)*h;
	t = np.linspace(t0, tn, x.shape[0]);

	plt.clf()	# clear figure potentially already open

	plt.plot(t,x)
	plt.xlabel('t [s]')
	plt.ylabel(ylabel)
	plt.title(f'Step size: {h}')
	plt.show()

def plotODEsolVar(x : Array[float], t0 : float, h : Array[float], ylabel : str = 'x(t)') -> None:
	"""Function plotting a solution of an ODE
		with variable stepsize.

	- **parameters**, **types**, **return** and **return types**::
		:param x: array containing ODE solution at time i x[i]
		:param t0: initial time
		:param h: step size
		:param ylabel: ylabel (default to x(t))
		:type x: np.array[float]
		:type t0: np.float
		:type h: np.float
		:type ylabel: string
		:return: None
		:rtype: None

	"""
	t = t0 + np.cumsum(h);

	plt.clf()	# clear figure potentially already open

	plt.plot(t,x)
	plt.xlabel('t [s]')
	plt.ylabel(ylabel)
	plt.title(f'Solution with variable stepsize.')
	plt.show()


def ODEphaseplot(x1 : Array[float], x2 : Array[float], t0 : float, h : float, xlabel : str = 'x_1', ylabel : str = 'x_2') -> None:
	"""Function plotting a phase portrait (without arrows) of an ODE solution.

	- **parameters**, **types**, **return** and **return types**::
		:param x1: array containing first component of ODE solution at time i x[i]
		:param x2: array containing second component of ODE solution at time i x[i]
		:param t0: initial time
		:param h: step size
		:param xlabel: xlabel (default to x_1)
		:param ylabel: ylabel (default to x_2)
		:type x1: np.array[float]
		:type x2: np.array[float]
		:type t0: np.float
		:type h: np.float
		:type xlabel: string
		:type ylabel: string
		:return: None
		:rtype: None

	"""

	plt.clf()	# clear figure potentially already open

	plt.plot(x1,x2)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(f'Step size: {h}')
	plt.show()
