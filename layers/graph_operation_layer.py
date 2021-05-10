import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

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

	def forward(self, x, A):
		assert A.size(1) == self.kernel_size
		x = self.conv(x)
		mask = A[:,7:]
		# A is (n,8,v,v)
		A = self.adjmatder(A[:,:7])
		########TODO##############
		'''
			1) MLP on the features
			2) Graph Conv - Gated GCN
			3) Edge Conv
			4) Node MLP post message passing
			5) Edge MLP post message passing
		'''

		edge_attr_list = []
		for i in range(A.shape[0]):
			index = torch.nonzero(mask[i],as_tuple=True)
			index = torch.vstack(index)
			edge_attr_list.append(torch.tensor([A[i,:7, index[0][m] , index[1][m] ] for m in range(index.shape[1])]))

		'''
		edge_attr_list contains the edge attributes for all the samples
	 	len(edge_attr_list) = A.shape[0] or n
		each element of edge_attr_list is the edge attributes tensor for that sample
		'''
		########TODO##############
		# A is (n,64,v,v)
		A = A*mask

		#Dl = ((A.sum(axis=2) + 0.001)**(-1)).float()
		# Dl is (n,64,v)
		#A = torch.einsum('ncvw,ncw->ncvw',(A,Dl))
		
		# To increase the no of channels of the graph to out_channels*k
		n, c, t, v = x.size()
		# x is now a 5d tensor with size (N,k,c,t,v)
		x = x.view(n, c, t, v)
		#A is (n,64,v,v)
		# Matrix multiplication followed by addition of all individual kernel matrices
		#print(x.shape)
		#print(A.shape)
		x = torch.einsum('nctv,ncvw->nctw', (x, A))

		return x.contiguous(), A
