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

# $Last changed: 2010 Feb 27$
# Developed since: May 2009
# File version: 0.2.1b

import numpy
import math
import copy

def make_meshgrid(Positions):
	dimension=len(Positions)
	Mesh=[]
	shape=map(len,Positions)
	for coordinatei,Position,coordinateshape in \
			zip(range(0,dimension),Positions,shape):
		Coordinate=numpy.asarray(Position)
		
		rawshape=numpy.ones(dimension)
		rawshape[coordinatei]=coordinateshape
		Coordinate=Coordinate.reshape(rawshape)

		for repeati,repeatdim in zip(range(0,dimension),shape):
			if repeati!=coordinatei:
				Coordinate=Coordinate.repeat(repeatdim,axis=repeati)

		Mesh.append(Coordinate)
	return Mesh

class GDFT:
	def __init__(self,Array,offsetK=None,offsetN=None,mode=None, 
			asymmetric = False):
		if mode is None:
			mode='GDFT'
		self.Array=Array
		dimension=len(self.Array.shape)
		self.shape=numpy.asarray(self.Array.shape,dtype=numpy.float)

		if offsetK is None:
			offsetK=numpy.zeros(dimension)
		elif offsetK=='centered':
			offsetK=-(self.shape-numpy.ones(dimension))/2
		else:
			offsetK=numpy.asarray(offsetK,dtype=numpy.float)

		if offsetN is None:
			offsetN=numpy.zeros(dimension)
		elif offsetN=='centered':
			offsetN=-(self.shape-numpy.ones(dimension))/2
		else:
			offsetN=numpy.asarray(offsetN,dtype=numpy.float)

		self.offsetK=offsetK
		self.offsetN=offsetN

		if mode not in ['GDFT','iGDFT']:
			raise ValueError('mode must be GDFT or iGDFT')
		self.mode=mode

		Ns=[]
		Ks=[]
		for coordinatei,coordinateshape in \
				zip(range(0,dimension),self.shape):
			baserange=numpy.arange(0,coordinateshape,dtype=numpy.float)

			N=baserange+self.offsetN[coordinatei]
			K=baserange+self.offsetK[coordinatei]

			Ns.append(N)
			Ks.append(K)

		self.MeshN=make_meshgrid(Ns)
		self.MeshK=make_meshgrid(Ks)

		if not asymmetric:
			self.normalisation=1/math.sqrt(self.shape.prod())
		elif mode == 'GDFT':
			self.normalisation = 1.0
		elif mode == 'iGDFT':
			self.normalisation = 1.0 / self.shape.prod()

	def get_by_K(self,K):
		K=copy.copy(K)
		K=numpy.asarray(K,dtype=numpy.float)
		K+=self.offsetK
		K/=self.shape
		Phasors=[]
		for N,k,offsetn in zip(self.MeshN,K,self.offsetN):
			Phasors.append((N+offsetn)*k)
		Phasor=sum(Phasors)
		Complex=numpy.exp(2*math.pi*complex(0,1)*Phasor)
		fourier_transform=self.normalisation*(self.Array*Complex).sum()
		return fourier_transform

	def get_by_N(self,N):
		N=copy.copy(N)
		N=numpy.asarray(N,dtype=numpy.float)
		N+=self.offsetN
		N/=self.shape
		Phasors=[]
		for K,n,offsetk in zip(self.MeshK,N,self.offsetK):
			Phasors.append(-(K+offsetk)*n)
		Phasor=sum(Phasors)
		Complex=numpy.exp(2*math.pi*complex(0,1)*Phasor)
		fourier_transform=self.normalisation*(self.Array*Complex).sum()
		return fourier_transform

	def get_by_kPositions(self,kPositions):
		def iterate(iterated_Position,iterate_Positions):
			if len(iterate_Positions)==0:
				return self.get_by_K(iterated_Position)
			else:
				remaining_Positions=iterate_Positions[1:]
				iterate_Position=iterate_Positions[0]
				return numpy.asarray(
						[iterate(
							iterated_Position+[position],
							remaining_Positions) for
						position in iterate_Position])
		Fourier_Transform=iterate([],kPositions)
		return Fourier_Transform

	def get_by_nPositions(self,nPositions):
		def iterate(iterated_Position,iterate_Positions):
			if len(iterate_Positions)==0:
				return self.get_by_N(iterated_Position)
			else:
				remaining_Positions=iterate_Positions[1:]
				iterate_Position=iterate_Positions[0]
				return numpy.asarray(
						[iterate(
							iterated_Position+[position],
							remaining_Positions) for
						position in iterate_Position])
		iFourier_Transform=iterate([],nPositions)
		return iFourier_Transform

	def get(self):
		Positions=[]
		for coordinateshape in self.Array.shape:
			Position=numpy.arange(0,coordinateshape,dtype=numpy.float)
			Positions.append(Position)

		if self.mode=='GDFT':
			return self.get_by_kPositions(Positions)
		elif self.mode=='iGDFT':
			return self.get_by_nPositions(Positions)
