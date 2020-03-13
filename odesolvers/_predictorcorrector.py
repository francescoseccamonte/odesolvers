#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Internal implementation of the predictor-corrector method of order 2, using Adams-Bashforth and Adams-Moulton.
#

import numpy as np

def _PECE_step(f, xi, ti, h, fpast, hpast, hpred, ETOL):
	"""Internal function implementing one step of the Predictor-Corrector
		linear multistep method.

	The ODE to be solved is of the form: x' = f(t,x), x being a vector in n-dimensions

	- **parameters**, **types**, **return** and **return types**::
		:param f: function in x' = f(t,x)
		:param xi: initial condition at time ti
		:param ti: current time
		:param h: step size
		:param fpast: previous function evaluations
		:param hpast: previous step sizes
		:param hpred: predicted step size to be used
		:param TOL: Error tolerance
		:type f: Callable
		:type xi: np.array[float]
		:type ti: np.float
		:type h: np.float
		:type fpast: RingBuffer(np.array[float])
		:type hpast: Ringbuffer(float)
		:type hpred: np.float
		:type TOL: np.float
		:return: Vector x containing solution of component j at next time ti+h (x[j]),
					corresponding stepsize hi, guessed stepsize for following iteration hfuture
		:rtype: np.array[float], float, float

	"""
	if xi.size > 1:
		fpast0 = fpast[0,:];
		fpast1 = fpast[1,:];
	else:
		fpast0 = fpast[0];
		fpast1 = fpast[1];

	if h is not None:		# fixed stepsize: no error checking
		# Predictor: AB2
		yp = xi + h*fpast1 + ((fpast1 - fpast0)/hpast[0])*h*h*0.5;
		# Corrector: AM2
		yc = xi + h*0.5*(f((ti+h), yp) + fpast1);

		return yc, h, h;
	else:					# adaptive stepsize
		hstep = hpred;
		lte : np.float = 0.0;	# initializing estimate of local truncation error
		while True:				# do-while loop
			# Predictor: AB2
			yp = xi + hstep*fpast1 + ((fpast1 - fpast0)/hpast[0])*hstep*hstep*0.5;
			# Corrector: AM2
			yc = xi + hstep*0.5*(f((ti+hstep), yp) + fpast1);
			lte = (5/6*np.linalg.norm(yc - yp));
			if hstep*lte <= ETOL:
				break			# step accepted, exiting loop
			else:
				hstep *= np.power(0.9*ETOL/(hstep*lte), 1.0/3.0);

		if lte <= 0.01*ETOL:
			hfuture = 2*hstep;		# doubling stepsize for following iteration
		else:
			hfuture = hstep;		# same stepsize for following iteration

		return yc, hstep, hfuture;
