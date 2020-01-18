#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Test Explicit Euler solver.
#

import unittest
import numpy as np

import odesolvers

def constantode(t,x):
	"""Function containing a constant ODE x' = 1.
	"""
	xprime = np.empty([1], float);
	xprime[0] = 1;
	return xprime;

def stableode(t,x):
	"""Function containing the ODE x' = -x.
	"""
	xprime = np.empty([1], float);
	xprime[0] = -x[0];
	return xprime;

class TestExplicitEuler(unittest.TestCase):
	def setUp(self):
		# common initial and final time for all tests
		self.t0 : np.float = 0.0;
		self.tn : np.float = 1.0;

	def testErrorHandling(self):
		iv = np.array([0.0]);
		with self.assertRaises(ValueError): odesolvers.ExplicitEulerSolver(constantode, iv, self.t0, self.tn, -0.1);
		with self.assertRaises(ValueError): odesolvers.ExplicitEulerSolver(constantode, iv, self.t0, self.tn, 0.0);
		with self.assertRaises(ValueError): odesolvers.ExplicitEulerSolver(constantode, iv, self.tn, self.t0, 0.1);


	def testConstantODE1(self):
		iv = np.array([0.0]);
		h : np.float = 0.1;

		y = odesolvers.ExplicitEulerSolver(constantode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0],iv);
		self.assertEqual(y[1],(y[0] + h));


	def testConstantODE2(self):
		iv = np.array([1.0]);
		h : np.float = 0.1;

		y = odesolvers.ExplicitEulerSolver(constantode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0],iv);
		self.assertEqual(y[1],(y[0] + h));

	def testStableODE1(self):
		iv = np.array([1.0]);
		h : np.float = 0.1;

		y = odesolvers.ExplicitEulerSolver(stableode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0],iv);
		self.assertEqual(y[1],(y[0] - h));

	def testStableODE2(self):
		iv = np.array([1.0]);
		h : np.float = 0.01;

		y = odesolvers.ExplicitEulerSolver(stableode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0],iv);
		self.assertEqual(y[1],(y[0] - h));


if __name__ == '__main__':
	unittest.main()
