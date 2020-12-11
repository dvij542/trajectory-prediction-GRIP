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
        """Constructor for ConvTemporalGraphical class.
        Arguments:
            in_channels {int} -- Number of input channels,
            out_channels {int} -- Number of output channels,
            kernel_size {int} -- Kernel size for increasing no of channels,
            t_kernel_size {int} -- Kernel size dimension,
            t_stride {int} -- Stride dimension,
            t_padding {int} -- Padding dimension,
            t_dilation {int} -- Dilation dimension,
            bias {bool} -- If True, adds a learnable bias to the output.

        """
        super().__init__()

        self.kernel_size = kernel_size
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
        """Forward function of the Graph operation layer.
        Arguments:
            x {torch.Tensor} -- Input to the layer -- [n (batch_size), 
                                                        kc (kernel size * channels), 
                                                        t (time_step), 
                                                        v (nodes)]
            A {torch.Tensor} -- Fixed Graph (Adjacency Matrix) -- [n (batch_size),
                                                                    k (kernel_size),
                                                                    v (nodes),
                                                                    w (nodes)]
        
        Returns:
        (torch.Tensor, torch.Tensor) -- (Output to the layer, Adjacency Matrix)
        """
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        # To increase the no of channels of the graph to out_channels*k
        n, kc, t, v = x.size()
        # x is now a 5d tensor with size (N,k,c,t,v)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        # Matrix multiplication followed by addition of all individual kernel matrices
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x.contiguous(), A
