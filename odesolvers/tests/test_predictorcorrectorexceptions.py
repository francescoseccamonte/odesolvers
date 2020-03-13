#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Test Predictor Corrector method Exceptions.
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

class TestAB_AM_PECE2Exceptions(unittest.TestCase):
	def setUp(self):
		# common initial and final time for all tests
		self.t0 : np.float = 0.0;
		self.tn : np.float = 1.0;

	def testErrorHandling(self):
		iv = np.array([0.0]);
		# negative step size
		with self.assertRaises(ValueError): odesolvers.AB_AM_PECE2(stableode, iv, self.t0, self.tn, -0.1);
		# zero time step
		with self.assertRaises(ValueError): odesolvers.AB_AM_PECE2(stableode, iv, self.t0, self.tn, 0.0);
		# time flowing negatively
		with self.assertRaises(ValueError): odesolvers.AB_AM_PECE2(stableode, iv, self.tn, self.t0, 0.1);
		# Negative numerical tolerance
		with self.assertRaises(ValueError): odesolvers.AB_AM_PECE2(stableode, iv, self.t0, self.tn, 0.1, ETOL=-0.1);


if __name__ == '__main__':
	unittest.main()
