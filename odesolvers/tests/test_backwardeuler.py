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

class TestBackwardEuler(unittest.TestCase):
	def setUp(self):
		# common initial and final time for all tests
		self.t0 : np.float = 0.0;
		self.tn : np.float = 0.3;


	def testConstantODE1(self):
		iv = np.array([0.0]);
		h : np.float = 0.1;

		y = odesolvers.ImplicitEulerSolver(constantode, iv, self.t0, self.tn, h, constantodeJ);

		self.assertEqual(y[0],iv);
		self.assertGreaterEqual(y[1], y[0]);


	def testConstantODE2(self):
		iv = np.array([1.0]);
		h : np.float = 0.1;

		y = odesolvers.ImplicitEulerSolver(constantode, iv, self.t0, self.tn, h, constantodeJ);

		self.assertEqual(y[0],iv);
		self.assertGreater(y[1], y[0]);

	def testStableODE1(self):
		iv = np.array([1.0]);
		h : np.float = 0.1;

		y = odesolvers.ImplicitEulerSolver(stableode, iv, self.t0, self.tn, h, stableodeJ);

		self.assertEqual(y[0],iv);
		self.assertLess(y[1], y[0]);

	def testStableODE2(self):
		iv = np.array([1.0]);
		h : np.float = 0.01;

		y = odesolvers.ImplicitEulerSolver(stableode, iv, self.t0, self.tn, h, stableodeJ);

		self.assertEqual(y[0],iv);
		self.assertLess(y[1], y[0]);

	def testMultivariableODE(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.01;

		y = odesolvers.ImplicitEulerSolver(multivariableode, iv, self.t0, self.tn, h, multivariableodeJ);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(y[1,1], y[0,1]);

	def testStiffODE1(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.01;

		N : np.uint = np.uint(np.ceil((self.tn - self.t0)/h));	# final step
		y = odesolvers.ImplicitEulerSolver(stiffode, iv, self.t0, self.tn, h, stiffodeJ);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(np.absolute(y[N,1]-y[0,1]),3);

	def testStiffODE2(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.05;

		N : np.uint = np.uint(np.ceil((self.tn - self.t0)/h));	# final step
		y = odesolvers.ImplicitEulerSolver(stiffode, iv, self.t0, self.tn, h, stiffodeJ);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(np.absolute(y[N,1]-y[0,1]),3);

	def testNewtonItConvergenceFail(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.01;

		with self.assertRaises(ArithmeticError): odesolvers.ImplicitEulerSolver(multivariableode, iv, self.t0, self.tn, h, multivariableodeJ, NEWTITER=0);

if __name__ == '__main__':
	unittest.main()
