#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Test function helpers.
#

import numpy as np

def constantode(t,x):
	"""Function containing a constant ODE x' = 1.
	"""
	xprime = np.empty([1], float);
	xprime[0] = 1;
	return xprime;

def constantodeJ(t, x):
	"""Function containing the Jacobian of constantode.
	"""

	df = np.empty([1,1], float);
	
	df[0,0] = 0;
	
	return df;

def stableode(t,x):
	"""Function containing the ODE x' = -x.
	"""
	xprime = np.empty([1], float);
	xprime[0] = -x[0];
	return xprime;

def stableodeJ(t, x):
	"""Function containing the Jacobian of stableode.
	"""

	df = np.empty([1,1], float);
	
	df[0,0] = -1;
	
	return df;

def multivariableode(t,x):
	"""Function containing the ODE 	x_1' = -x_1 + x_2
									x_2' = -x_2 .
	"""
	xprime = np.empty([2], float);
	xprime[0] = -x[0] + x[1];
	xprime[1] = -x[1];
	return xprime;

def multivariableodeJ(t, x):
	"""Function containing the Jacobian of multivariableode.
	"""

	df = np.empty([2,2], float);
	
	df[0,0] = -1;
	df[0,1] = +1;
	df[1,0] = 0;
	df[1,1] = -1;
	
	return df;

def stiffode(t, x):
	"""Function containing the stiff ODE 	x_1' = -x_1
											x_2' = -100(x_2 - sin(t)) + cos(t).
	"""
	xprime = np.empty([2], float);

	xprime[0] = -x[0];
	xprime[1] = -100*(x[1] - np.sin(t)) + np.cos(t);

	return xprime;

def stiffodeJ(t, x):
	"""Function containing the Jacobian of stiffode.
	"""

	df = np.empty([2,2], float);
	
	df[0,0] = -1;
	df[0,1] = 0;
	df[1,0] = 0;
	df[1,1] = -100;
	
	return df;
