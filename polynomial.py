# Copyright (c) 2010 Friedrich Romstedt <www.friedrichromstedt.org>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Last changed: 2010 Feb 27
# Developed since: Feb 2010
# File version: 0.1.0b

import numpy
import gdft

"""Semi-fast polynomial multiplication."""


class Polynomial:
	"""Read-only arguments:

	.coefficients - The coefficients."""
	def __init__(self, coefficients = None):
		"""Initialise from real-space COEFFICIENTS."""

		self.coefficients = numpy.asarray(coefficients)

	def __mul__(self, other):
		"""OTHER is supposed to be a Polynomial too."""
		
		# Find out the rank of the result ...

		rank_result = (len(self.coefficients) - 1) + \
				(len(other.coefficients) - 1) + 1

		# Pad the coefficients with zeros ...

		self_padded = numpy.zeros(rank_result)
		self_padded[:len(self.coefficients)] = self.coefficients

		other_padded = numpy.zeros(rank_result)
		other_padded[:len(other.coefficients)] = other.coefficients

		# Perform DFT ...

		self_gdft = gdft.GDFT(self_padded, asymmetric = True)
		self_dft = self_gdft.get()

		other_gdft = gdft.GDFT(other_padded, asymmetric = True)
		other_dft = other_gdft.get()

		# Calculate the Fourier transform of the result ...

		dft12 = self_dft * other_dft

		# Transform back ...

		igdft = gdft.GDFT(dft12, mode = 'iGDFT', asymmetric = True)
		coefficients12 = igdft.get()

		return Polynomial(coefficients12)
	
	def __str__(self):
		return str(self.coefficients)
