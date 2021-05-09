import torch
import torch.nn as nn
import numpy as np
import os 
CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
dev = 'cuda:0' 


class ConvTemporalGraphical(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size,
				 t_kernel_size=1,
				 t_stride=1,
				 t_padding=0,
				 t_dilation=1,
				 bias=True):
		super().__init__()
		self.x1 = None
		self.x2 = None

		self.kernel_size = kernel_size
		self.adjmatder = nn.Sequential(
				nn.Conv2d(
					7,
					16,
					kernel_size = 1,
					stride=(1,1)),
				#nn.BatchNorm2d(16),
				nn.ReLU(inplace=False),
				nn.Dropout(0.5, inplace=False),
				nn.Conv2d(
					16,
					32,
					kernel_size = 1,
					stride=(1,1)),
				#nn.BatchNorm2d(16),
				nn.ReLU(inplace=False),
				nn.Dropout(0.5, inplace=False),
				nn.Conv2d(
					32,
					64,
					kernel_size = 1,
					stride=(1,1)),
				#nn.BatchNorm2d(16),
				#nn.BatchNorm2d(16),
				nn.ReLU(inplace=False),
				nn.Dropout(0.5, inplace=False)

			)
		# To increase the no of channels of the graph to out_channels*k
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=(t_kernel_size, 1),
			padding=(t_padding, 0),
			stride=(t_stride, 1),
			dilation=(t_dilation, 1),
			bias=bias)
		self.l1 = nn.Linear(2,1)
		self.l2 = nn.Linear(6,1)
		self.l3 = nn.Linear(2,1)
		self.relu= nn.LeakyReLU(0.2)
		self.softmax = nn.Softmax2d()


	def forward(self, x, A):
		assert A.size(1) == self.kernel_size
		x = self.conv(x)
		mask = A[:,7:]

		A = self.adjmatder(A[:,:7])

		# A is (n,64,v,v)
		#A = A*mask
		#Dl = ((A.sum(axis=2) + 0.001)**(-1)).float()
		# Dl is (n,64,v)
		#A = torch.einsum('ncvw,ncw->ncvw',(A,Dl))
		
		# To increase the no of channels of the graph to out_channels*k
		n, c, t, v = x.size()
		# x is now a 5d tensor with size (N,k,c,t,v)
		x1 = x.view(n*c*v, t)
		x1 = x1.to(dev)
		x1 = self.l2(x1)
		x1 = x1.view(n,c,v)
		x2 = torch.ones([n, c, v, v], dtype=torch.float64)
		x2 = x2.to(dev)
		for i in range(0, v):
			x2[: ,: ,i, :] = x1
		A = self.relu(x2)

		#print(x1.shape)
		

		A = self.softmax(A)
		A = A.float().to(dev)
		A = A*mask
		#A is (n,64,v,v)
		# Matrix multiplication followed by addition of all individual kernel matrices
		#print(x.shape)
		#print(A.shape)
		x = torch.einsum('nctv,ncvw->nctw', (x, A))

		return x.contiguous(), A

