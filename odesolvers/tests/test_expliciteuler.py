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

from .test_helpers import *

class TestExplicitEuler(unittest.TestCase):
	def setUp(self):
		# common initial and final time for all tests
		self.t0 : np.float = 0.0;
		self.tn : np.float = 0.3;


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

	def testMultivariableODE(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.01;

		y = odesolvers.ExplicitEulerSolver(multivariableode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertEqual(y[1,0],(iv[0]+h*(-iv[0]+iv[1])));
		self.assertEqual(y[1,1],(iv[1]-h*iv[1]));

	def testStiffODE1(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.01;

		N : np.uint = np.uint(np.ceil((self.tn - self.t0)/h));	# final step
		y = odesolvers.ExplicitEulerSolver(stiffode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(np.absolute(y[N,1]-y[0,1]),3);

	def testStiffODE2(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.05;

		N : np.uint = np.uint(np.ceil((self.tn - self.t0)/h));	# final step
		y = odesolvers.ExplicitEulerSolver(stiffode, iv, self.t0, self.tn, h);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertGreater(np.absolute(y[N,1]-y[0,1]),3);	# stepsize not small enough to solve stiff equation


if __name__ == '__main__':
	unittest.main()
