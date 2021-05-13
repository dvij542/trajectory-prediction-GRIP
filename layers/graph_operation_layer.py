import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
dev = 'cuda:0'
#CUDA_LAUNCH_BLOCKING=1


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
		x_y = x.clone()
		mask = A[:,7:]
		#print(mask.shape)

		#A = self.adjmatder(A[:,:7])
		A = A[:,:7]
		x1 = x_y.permute(0,3,1,2).clone()
		#print(x1.shape)
		n, v, c, t = x1.size()
		x8 = x1.reshape(n*v*c,t).clone()
		x8 = x8.to(dev)
		#print(x1.shape)
		x9 = self.l2(x8)
		x10 = x9.reshape(n,v,c).clone()
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
			x2[: ,: ,i, :] = x10.clone()
			x3[: ,i ,:, :] = x10.clone()
		x4 = torch.cat((x2,x3),dim=-1)
		x5 = x4.reshape(n*v*v,128).clone()
		#print(x3.shape)
		x6 = x5.float().to(dev)
		x7 = self.l1(x6)
		A1 = self.relu(x7)
		#print(A1)
		A2 = A1.reshape(n,v,v,1).clone()
		mask1 = mask.permute(0,2,3,1).clone()
		mask2 = mask1.long()
		#mask = mask.reshape(n,v,v)
		#A1 = A1.reshape(n,v,v)
		#print(mask2.shape)
		#print(A2.shape)
		#print(A1)
		A2[mask2==0] = float(-1000)
		#print(mask[1,: ,:])
		#print(A1[1,:,:])
		    
		    
		A3 = A2.reshape(n,v,v).clone()
		#for i in range(0, n):
			
		A4 = self.softmax(A3)
		#p = A1
		#A4[A4!=A4] = 0
		#print(A4)
		#print(A1)
		A5 = A4.reshape(n,v,v,1).clone()
		A6 = A5.permute(0,3,1,2).clone()
		n, d, a, b = A6.size()


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
		x_y1 = torch.einsum('nctv,ndvw->ncdtw', (x_y.clone(), A6.clone())).clone()
		x = x_y1.clone().reshape(n,c,t,v)
		#A1 = A1.detach()
		print(x.shape)
		#print(x)

		return x.contiguous(), A6



