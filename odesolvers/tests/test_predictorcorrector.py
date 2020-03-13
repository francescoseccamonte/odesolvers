# Author : Francesco Seccamonte
# Copyright (c) 2020 Francesco Seccamonte. All rights reserved.  
# Licensed under the MIT License. See LICENSE file in the project root for full license information.  
#

#
# Test predictor corrector method AB-AM2.
#

import unittest
import numpy as np

import odesolvers

from .test_helpers import *

class TestAB_AM_PECE2(unittest.TestCase):
	def setUp(self):
		# common initial and final time for all tests
		self.t0 : np.float = 0.0;
		self.tn : np.float = 0.2;
		self.h : np.float = 0.01;

	def testSimpleODE(self):
		iv = np.array([1.0]);

		# test fixed stepsize
		y, hi = odesolvers.AB_AM_PECE2(stableode, iv, self.t0, self.tn, self.h);

		self.assertEqual(y[0],iv);
		self.assertEqual(y[1],(y[0] - self.h));
		self.assertLess(y[20],y[0]);

		# test adaptive stepsize
		y, hi = odesolvers.AB_AM_PECE2(stableode, iv, self.t0, self.tn, h=None);

		self.assertEqual(y[0],iv);
		self.assertLess(y[1],y[0]);
		self.assertLess(y[20],y[0]);

	def testMultivariableODE(self):
		iv = np.array([1.0, 2.0]);

		# test fixed stepsize
		y, hi = odesolvers.AB_AM_PECE2(multivariableode, iv, self.t0, self.tn, self.h);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertEqual(y[1,1],(y[0,1] - iv[1]*self.h));
		self.assertLess(y[20,1],y[0,1]);

		# test adaptive stepsize
		y, hi = odesolvers.AB_AM_PECE2(multivariableode, iv, self.t0, self.tn, h=None);

		self.assertEqual(y[0,0],iv[0]);
		self.assertEqual(y[0,1],iv[1]);
		self.assertLess(y[1,1],y[0,1]);
		self.assertLess(y[20,1],y[0,1]);

