import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
dev = 'cuda:0'
CUDA_LAUNCH_BLOCKING=1


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
		self.l1 = nn.Linear(128,1)
		self.l2 = nn.Linear(6,1)
		#self.l3 = nn.Linear(2,1)
		#self.relu= nn.LeakyReLU(0.2)
		self.softmax = nn.Softmax(dim=2)
		
		
		# LeakyReLU
		self.relu = nn.LeakyReLU(0.2,inplace=False)


	def forward(self, x, A):
		assert A.size(1) == self.kernel_size
		x = self.conv(x)
		mask = A[:,7:]
		#print(mask.shape)

		#A = self.adjmatder(A[:,:7])
		A = A[:,:7]
		x1 = x.permute(0,3,1,2)
		#print(x1.shape)
		n, v, c, t = x1.size()
		x1 = x1.reshape(n*v*c,t)
		x1 = x1.to(dev)
		#print(x1.shape)
		x1 = self.l2(x1)
		x1 = x1.reshape(n,v,c)
		#x1 = self.l1(x1)
		#x1 = x1.reshape(n, v, 7)s
		#A = A.permute(0,2,3,1)
		#x2 = torch.matmul(torch.transpose(x1,0,1),x1)
		x2 = torch.ones([n, v, v, c], dtype=torch.float64)
		#print(x2.shape)
		x3 = torch.ones([n, v, v, c], dtype=torch.float64)
		x2 = x2.to(dev)
		x3 = x3.to(dev)
		for i in range(0, v):
			x2[: ,: ,i, :] = x1
			x3[: ,i ,:, :] = x1
		x3 = torch.cat((x2,x3),dim=-1)
		x3 = x3.reshape(n*v*v,128)
		#print(x3.shape)
		x3 = x3.float().to(dev)
		x3 = self.l1(x3)
		A1 = self.relu(x3)
		#print(A1)
		A1 = A1.reshape(n,v,v,1)
		mask = mask.permute(0,2,3,1)
		mask = mask.long()
		#mask = mask.reshape(n,v,v)
		#A1 = A1.reshape(n,v,v)
		print(mask.shape)
		print(A1.shape)
		#print(A1)
		A1[mask==0] = float('-inf')
		#print(mask[1,: ,:])
		#print(A1[1,:,:])
		    
		    
		A1 = A1.reshape(n,v,v)
		#for i in range(0, n):
			
		A1 = self.softmax(A1)
		#p = A1
		A1[A1!=A1] = 0
		print(A1)
		#print(A1)
		A1 = A1.reshape(n,v,v,1)
		A1 = A1.permute(0,3,1,2)
		n, d, a, b = A1.size()


		#A = A.reshape(n,v,v,7)
		#A = A.permute(0,3,1,2)
		#A = self.adjmatder(A[:,:7])
		#x = self.conv(x)
		# A is (n,64,v,v)
		#A = A*mask#Dl = ((A.sum(axis=2) + 0.001)**(-1)).float()
		# Dl is (n,64,v)
		#A = torch.einsum('ncvw,ncw->ncvw',(A,Dl))
		
		# To increase the no of channels of the graph to out_channels*k
		

		#print(x1.shape)
		

		#A = self.softmax(A)
		#A = A*mask
		#A is (n,64,v,v)
		# Matrix multiplication followed by addition of all individual kernel matrices
		#print(x.shape)
		#print(A.shape)
		x = torch.einsum('nctv,ndvw->nctw', (x, A1))
		#A1 = A1.detach()
		print(x.shape)
		#print(x)

		return x.contiguous(), A1


