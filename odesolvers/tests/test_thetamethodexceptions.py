#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Test Theta method Exceptions.
#

import unittest
import numpy as np

import odesolvers

def stableode(t,x):
	"""Function containing the ODE x' = -x.
	"""
	xprime = np.empty([1], float);
	xprime[0] = -x[0];
	return xprime;

class TestThetaMethodExceptions(unittest.TestCase):
	def setUp(self):
		# common initial and final time for all tests
		self.t0 : np.float = 0.0;
		self.tn : np.float = 1.0;

	def testErrorHandling(self):
		iv = np.array([0.0]);
		# negative time step
		with self.assertRaises(ValueError): odesolvers.ThetaMethod(stableode, iv, self.t0, self.tn, -0.1, 1);
		# zero time step
		with self.assertRaises(ValueError): odesolvers.ThetaMethod(stableode, iv, self.t0, self.tn, 0.0, 1);
		# time flowing negatively
		with self.assertRaises(ValueError): odesolvers.ThetaMethod(stableode, iv, self.tn, self.t0, 0.1, 1);
		# theta out of bounds
		with self.assertRaises(ValueError): odesolvers.ThetaMethod(stableode, iv, self.t0, self.tn, 0.1, 2);
		# Negative numerical tolerance
		with self.assertRaises(ValueError): odesolvers.ThetaMethod(stableode, iv, self.t0, self.tn, 0.1, 0.5, TOL=-0.1);
		# Negative number of Newton iteration steps
		with self.assertRaises(ValueError): odesolvers.ThetaMethod(stableode, iv, self.t0, self.tn, 0.1, 0.5, NEWTITER=-2);
		# Automatic differentiation
		with self.assertRaises(NotImplementedError): odesolvers.ThetaMethod(stableode, iv, self.t0, self.tn, 0.1, 0);


if __name__ == '__main__':
	unittest.main()
