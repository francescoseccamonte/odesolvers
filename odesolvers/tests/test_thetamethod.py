#
# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Test Theta method (trapezoidal for theta = 0.5) solver.
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
		self.theta : np.float = 0.1;

	def testMultivariableODE(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.01;

		y = odesolvers.ThetaMethod(multivariableode, iv, self.t0, self.tn, h, self.theta, multivariableodeJ);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(y[1,1], y[0,1]);

	def testStiffODEConvergent(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.05;

		N : np.uint = np.uint(np.ceil((self.tn - self.t0)/h));	# final step
		y = odesolvers.ThetaMethod(stiffode, iv, self.t0, self.tn, h, self.theta, stiffodeJ);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(np.absolute(y[N,1]-y[0,1]),3);

	def testStiffODENotConvergent(self):
		iv = np.array([1.0, 2.0]);
		h : np.float = 0.05;

		N : np.uint = np.uint(np.ceil((self.tn - self.t0)/h));	# final step
		y = odesolvers.ThetaMethod(stiffode, iv, self.t0, self.tn, h, 0.8, stiffodeJ);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertGreater(np.absolute(y[N,1]-y[0,1]),3);
