import torch
import torch.nn as nn


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
                    5,
                    16,
                    kernel_size = 1,
                    stride=1),
                #nn.BatchNorm2d(16),
                nn.ReLU(inplace=False),
                nn.Dropout(0.5, inplace=False),
                nn.Conv2d(
                    16,
                    32,
                    kernel_size = 1,
                    stride=1),
                #nn.BatchNorm2d(16),
                nn.ReLU(inplace=False),
                nn.Dropout(0.5, inplace=False),
                nn.Conv2d(
                    32,
                    64,
                    kernel_size = 1,
                    stride=1),
                #nn.BatchNorm2d(16),
                #nn.BatchNorm2d(16),
                #nn.ReLU(inplace=False),
                nn.Dropout(0.5, inplace=False)

            )
        # To increase the no of channels of the graph to out_channels*k
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size=1,bias=bias)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        # To increase the no of channels of the graph to out_channels*k
        n, kc, t, v = x.size()
        mask1 = A[:,5:]
        mask2 = torch.zeros((n,1,v,v))
        for i in range(120) :
            mask1[:,0,i,i] = 0
            mask2[:,0,i,i] = 1
        A = self.adjmatder(A[:,:5])
        # A is (n,64,v,v)
        A = torch.tensor((A*mask1,A*mask2)).to('cuda:0')
        # A is (2,n,64,v,v)
        #Dl = ((A.sum(axis=3) + 0.001)**(-1)).float()
        # Dl is (n,64,v)
        #A = torch.einsum('ncvw,ncw->ncvw',(A,Dl))
        
        # x is now a 5d tensor with size (N,k,c,t,v)
        x = x.view(n, 2, kc/2, t, v)
        #A is (n,64,v,v)
        # Matrix multiplication followed by addition of all individual kernel matrices
        x = self.relu(torch.einsum('nkctv,kncvw->nctw', (x, A)))


        return x.contiguous(), A
