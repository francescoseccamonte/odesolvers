#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# File implementing the variable stepsize predictor-corrector method of order 2,
# using Adams-Bashforth and Adams-Moulton.
#

import numpy as np
from nptyping import Array
from typing import Tuple
from numpy_ringbuffer import RingBuffer

from ._expliciteuler import _ExplicitEuler_step
from ._predictorcorrector import _PECE_step

def AB_AM_PECE2(f, iv : Array[float], t0 : float, tn : float, h : float = None, ETOL : float = 1.0e-5) -> Tuple[Array[float], Array[float]]:
	"""Function implementing the predictor-corrector method of order 2, using Adams-Bashforth and Adams-Moulton.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param iv: vector of initial values
		:param t0: initial time
		:param tn: final time
		:param h: step size
		:param ETOL: Error tolerance
		:type f: Callable
		:type iv: np.array[float]
		:type t0: np.float
		:type tn: np.float
		:type h: np.float
		:type TOL: np.float
		:return: Vector x containing solution of component j at time i (x[i,j]), and corresponding stepsizes hi
		:rtype: np.array[float,float], np.array[float]

	"""

	if h is not None and h <= 0.0:
		raise ValueError('The stepsize h must be positive')

	if (tn - t0) <= 0.0:
		raise ValueError('The final time must be greater than the initial time')

	if ETOL <= 0.0:
		raise ValueError('The numerical tolerance must be positive')

	# circular buffers to store previous function evaluations and stepsizes
	fprevAB = RingBuffer(2, dtype=((np.float, iv.size) if iv.size > 1 else np.float));
	hprev = RingBuffer(1, dtype=np.float);		# -1 wrt fprevAB size

	N = np.int(np.ceil((tn - t0)/(h if h is not None else 0.01)));	# number of steps (guess in case h is None)
	# x collects the states, hi the corresponding stepsizes
	# preallocating the x and hi arrays (+1 for including initial condition)
	x = np.empty((np.int(N+1),iv.size), float);
	hi = np.empty((np.int(N+1),1), float);

	x[0,:] = iv;
	hi[0] = 0.0;

	# first point after Initial Value is obtained through explicit euler method
	if h is None:
		hi[1] = 0.01;	# stepsize not provided, starting at 0.01
	else:
		hi[1] = h;

	x[1,:] = _ExplicitEuler_step(f,x[0,:],t0,hi[1]);
	fprevAB.append(f(t0,x[0,:]));
	fprevAB.append(f(t0+hi[1],x[1,:]));
	hprev.append(hi[1]);

	tcount = t0 + hi[1];		# to check for termination
	i : np.uint = 2;			# to check for vector sizes
	hfuture : np.float = hi[1];	# guess on future stepsize (updated at every iteration)

	while tcount < tn:
		if i >= N:
			N *= 2;
			x = np.resize(x, (N,iv.size));
			hi = np.resize(hi, (N,1));

		x[i,:], hi[i], hfuture = _PECE_step(f,x[i-1,:],tcount,h,fprevAB,hprev,hfuture,ETOL);
		tcount += hi[i];
		fprevAB.append(f(tcount,x[i,:]));
		hprev.append(hi[i]);
		i += 1;

	return x[:i,:], hi[:i];


def AB_AM_PECE2_interpatT(f, t : float, tvec : Array[float], xvec : Array[float]) -> Array[float]:
	"""Function computing numerical solution with the predictor-corrector
		method of order 2 at (potentially) off-step point t.

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param t: desired time
		:param tvec: vector of times where the solution is available
		:param xvec: solution at times tvec
		:type f: Callable
		:type t: np.float
		:type tvec: np.array[float]
		:type xvec: np.array[float]
		:return: Vector x containing solution at desired time
		:rtype: np.array[float]

	"""
	if (tvec[0] > t or tvec[-1] < t):
		print(f"t: {t}, tvec[-1]: {tvec[-1]}");
		raise ValueError('Error: t must lie in the interval t0, tn.')

	i = np.argwhere(tvec <= t)[-1];
	h = t - tvec[i].item();

	if (h < 1e-6):
		return xvec[i,:];

	if (i == 0):
		return xvec[i,:];

	# TODO compute previous function evaluations
	fprevAB = RingBuffer(2, dtype=((np.float, xvec.shape[1]) if xvec.shape[1] > 1 else np.float));
	hprev = RingBuffer(1, dtype=np.float);		# -1 wrt fprevAB size
	fprevAB.append(f(tvec[i-1],xvec[i-1,:].flatten()));
	fprevAB.append(f(tvec[i],xvec[i,:].flatten()));
	hprev.append(tvec[i]-tvec[i-1]);

	return _PECE_step(f,xvec[i,:].flatten(),tvec[i],h,fprevAB,hprev,None,None)[0];

