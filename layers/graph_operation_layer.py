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
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        mask = A[:,7]
        A = self.adjmatder(A[:,:7])
        A = A*mask
        Dl = ((A.sum(axis=2) + 0.001)**(-1)).float()
        A = torch.einsum('ncvw,ncw->ncvw',(A,Dl))
        
        # To increase the no of channels of the graph to out_channels*k
        n, c, t, v = x.size()
        # x is now a 5d tensor with size (N,k,c,t,v)
        x = x.view(n, c, t, v)
        #A is (n,64,v,v)
        # Matrix multiplication followed by addition of all individual kernel matrices
        x = torch.einsum('nctv,ncvw->nctw', (x, A))

        return x.contiguous(), A
