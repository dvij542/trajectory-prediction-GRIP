import torch
import torch.nn as nn
# import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
# import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
# from torch.nn import Parameter as Param
# from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv

import numpy as np


###################################old code with some changes :( ##########################################
class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.adjmatder = nn.Sequential(
            nn.Conv2d(
                7,
                16,
                kernel_size=1,
                stride=(1, 1)),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(
                16,
                32,
                kernel_size=1,
                stride=(1, 1)),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(
                32,
                64,
                kernel_size=1,
                stride=(1, 1)),
            # nn.BatchNorm2d(16),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5, inplace=False)
        )
        # To increase the no of channels of the graph to out_channels*k
        # self.conv = nn.Conv2d(
        # 	in_channels,
        # 	out_channels,
        # 	kernel_size=(t_kernel_size, 1),
        # 	padding=(t_padding, 0),
        # 	stride=(t_stride, 1),
        # 	dilation=(t_dilation, 1),
        # 	bias=bias)

        self.conv2 = anim_conv(in_channels, out_channels, kernel_size)

    def forward(self, x, A):
        assert A.size(1) == self.kernel_size
        # A is (n,8,v,v)
        mask = A[:, -1:]

        ########TODO##############
        '''
			1) MLP on the features
			2) Graph Conv - Gated GCN
			--------3) Edge Conv---------------- idtso
			4) Node MLP post message passing
			5) Edge MLP post message passing
		'''

        '''
			1) import MLP from GCN class
			2) GATconv with same input and output as Gated-GCN
		'''
        m, n = x.shape[0], x.shape[3]
        x_reshaped = x.reshape(m, -1, n)
        datalist = []
        for i in range(A.shape[0]):
            print(f"{mask[i].shape}")
            hello = torch.reshape(mask[i], (400, 400))
            index = torch.nonzero(hello, as_tuple=True)
            index = torch.vstack(index).detach().cpu()
            Ab = A.detach().cpu().numpy()
            edge_attr = torch.tensor(
                [Ab[i, :7, index[0][m], index[1][m]] for m in range(index.shape[1])])

            data = Data(x=torch.transpose(x_reshaped[i].detach(
            ).cpu(), 0, 1), edge_index=index, edge_attr=edge_attr)
            datalist.append(data)
            # print(data.x.shape)
        train_loader = DataLoader(
            datalist, batch_size=x.shape[0], shuffle=True)
        data1 = next(iter(train_loader)).to("cuda:0")
        # new_a = pyg_utils.to_dense_adj(
        # 	edge_index=data1.edge_index, batch=data1.batch, edge_attr=data1.edge_attr, max_num_nodes=400)
        # new_a = new_a.permute(0, 3, 1, 2).to('cuda:0')
        # new_a = torch.cat((new_a, mask), 1)
        ########TODO##############

        # x = self.conv(x)
        # print("starting conv2")
        x, A = self.conv2(data1, mask)  # CALL TO NEW CONV LAYER

        # include GAT with data1 and mask

        # print("YESSSS")
        # print(x.shape)
        x= x.reshape(A.shape[0], x.shape[0] //
                         A.shape[0], 64, 6).permute(0, 2, 3, 1)
        # print(x.shape)
        # A is (n,8,v,v)
        # A = self.adjmatder(A[:, :7])
        A = A*mask
        # print(A.shape)
        # Dl = ((A.sum(axis=2) + 0.001)**(-1)).float()
        # Dl is (n,64,v)
        # A = torch.einsum('ncvw,ncw->ncvw',(A,Dl))

        # To increase the no of channels of the graph to out_channels*k
        # ,64,6,400
        n, c, t, v = x.size()
        # x is now a 5d tensor with size (N,k,c,t,v)
        x = x.view(n, c, t, v)
        # A is (n,64,v,v)
        # Matrix multiplication followed by addition of all individual kernel matrices
        # print(x.shape)
        # print(A.shape)
        x = torch.einsum('nctv,ncvw->nctw', (x, A))

        return x.contiguous(), A


class anim_MLP(nn.Module):
    '''
    code taken from  https://github.com/dvl-tum/mot_neural_solver/blob/master/src/mot_neural_solver/models/mlp.py
    '''

    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(anim_MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))
        # print(type(input_dim),input_dim)
        layers = []
        for dim in fc_dims:
            # print(f"trying to append {input_dim,dim}")
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)


class anim_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size

        self.edge_in_dim = 7  # check this
        self.edge_fc_dims = [256, 512]  # check this
        self.edge_out_dim = 384  # check this
        self.edge_in_dim1 = 384  # check this
        self.edge_fc_dims1 = [512,256]  # check this
        self.edge_out_dim1 = 63  # check this
        self.edge_mid = [512]
        self.edge_update_in = 3*384
        if(in_channels == 4):
            self.node_in_dim = 24  # check this
        else:
            self.node_in_dim = 384
        self.node_fc_dims = [256, 512]  # check this
        if(in_channels == 4):
            self.node_out_dim = 24  # check this
        else:
            self.node_out_dim = 384
        self.dropout_p = 0.5  # checkthis
        self.training = True  # add an option for this

        self.edge_mlp = anim_MLP(input_dim=self.edge_in_dim, fc_dims=list(self.edge_fc_dims) + [self.edge_out_dim],
                                 dropout_p=self.dropout_p, use_batchnorm=True)
        self.edge_mlp1 = anim_MLP(input_dim=self.edge_in_dim1, fc_dims=list(self.edge_fc_dims1) + [self.edge_out_dim1],
        						 dropout_p=self.dropout_p, use_batchnorm=True)
        self.node_mlp = anim_MLP(input_dim=self.node_in_dim, fc_dims=list(self.node_fc_dims) + [self.node_out_dim],
                                 dropout_p=self.dropout_p, use_batchnorm=True)
        self.edge_update = EdgeUpdate(input_dim=self.edge_update_in, fc_dims=list(self.edge_mid) + [self.edge_out_dim],
                                      dropout_p=self.dropout_p, use_batchnorm=True)

        # print(f"inchannels and out channels: {in_channels},{out_channels}")
        # self.convs = pyg_nn.GatedGraphConv(out_channels=out_channels,num_layers=2)
        # self.convs = pyg_nn.GatedGraphConv(out_channels=384,num_layers=2)
        self.convs_n = GatedGraphConv_parv(out_channels=384, num_layers=2)

    def forward(self, data, mask):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        edge_features = self.edge_mlp(edge_attr)
        node_features = self.node_mlp(x)
        node_features = self.convs_n(
            x=node_features, edge_index=edge_index, edge_weight=edge_features)
        node_features = F.relu(node_features)
        node_features = F.dropout(
            node_features, p=self.dropout_p, training=self.training)

        # edge_features = self.edge_mlp1(edge_features)
        edge_features = self.edge_update(node_features,edge_index,edge_features)
        edge_features = self.edge_mlp1(edge_features)
        # A_new = from_edge_idx(edge_index, edge_features, batch)
        A_new = to_dense_adj(
            edge_index=edge_index, batch=batch, edge_attr=edge_features, max_num_nodes=400)
        A_new = A_new.permute(0, 3, 1, 2).to('cuda:0')
        # A_new = A_new * mask
        A_new = torch.cat((A_new, mask), 1)
        # print(node_features.shape,A_new.shape)
        return node_features, A_new
        # return node_features
        ####################

# possible template for implementing gated graph conv on own###################################################3


class EdgeUpdate(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        """ inputs = [x, edge_weight, edge_index]
        """
        super().__init__()
        # super().build(input_shape)
        print(type(fc_dims))
        self.gather = Gather
        self.slice1 = Slice(np.s_[1, :])
        self.slice0 = Slice(np.s_[0, :])
        self.concat = ConcatDense(input_dim=input_dim, fc_dims=fc_dims,
                                  dropout_p=dropout_p, use_batchnorm=use_batchnorm)

    def forward(self, x, edge_index, edge_weight, mask=None):
        """ Inputs: [x, edge_weight, edge_index]
                Outputs: edge_weight
        """

        # Get nodes at start and end of edge
        print(f"IN ERROR ZONE {edge_index.shape}")
        # source_edge_index = self.slice0.call(edge_index).transpose(0,1)
        # source_node = self.gather(x, self.slice0.call(edge_index))
        # source_node = x[source_edge_index]
        edge_index_T = edge_index.permute(1,0)
        x_numpy = x.detach().cpu().numpy()
        source_node=torch.tensor([x_numpy[y[0]] for y in edge_index_T]).to('cuda:0')
        # edge_attr = torch.tensor(
        #         [Ab[i, :7, index[0][m], index[1][m]] for m in range(index.shape[1])])
        target_node=torch.tensor([x_numpy[y[1]] for y in edge_index_T]).to('cuda:0')
        print(f"IN TESTING ZONE {source_node.shape} {target_node.shape}")

        # target_node = self.gather(x, self.slice1.call(edge_index))

        new_edge_weight = self.concat([edge_weight, source_node, target_node])

        # if self.dropout > 0.:
        #     new_edge_weight = self.dropout_layer(new_edge_weight)

        return new_edge_weight


class Slice():
    def __init__(self, slice_obj):
        # super(Slice, self).__init__(*args, **kwargs)
        self.slice_obj = slice_obj
        # self.supports_masking = True

    def call(self, inputs):
        return inputs[self.slice_obj]


def Gather(reference,indices):
    # def call(self, inputs, mask=None):
    # reference, indices = inputs
    return torch.gather(reference,0, indices)


class ConcatDense(nn.Module):
    '''
    code taken from  https://github.com/dvl-tum/mot_neural_solver/blob/master/src/mot_neural_solver/models/mlp.py
    '''

    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(ConcatDense, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        print(f"IN TESTING ZONE {input[0].shape} {input[1].shape} {input[2].shape}")

        input = torch.cat(input, dim=1)
        print(f"IN TESTING ZONE {input.shape}")

        return self.fc_layers(input)


class GatedGraphConv_parv(GatedGraphConv):
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super(GatedGraphConv_parv, self).__init__(out_channels=out_channels,
                                                  num_layers=num_layers, aggr=aggr, bias=bias, **kwargs)

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight * x_j

# class ConcatDense(layers.Layer):
# 	""" Layer to combine the concatenation and two dense layers. Just useful as a common operation in the graph
# 	layers """

# 	def __init__(self, **kwargs):
# 		super(ConcatDense, self).__init__(**kwargs)
# 		self.supports_masking = True

# 	def build(self, input_shape):
# 		num_features = input_shape[0][-1]
        # self.concat = layers.Concatenate()
        # self.dense1 = layers.Dense(2 * num_features, activation='relu')
        # self.dense2 = layers.Dense(num_features)

    # def call(self, inputs, mask=None):
    # 	output = self.concat(inputs)
    # 	output = self.dense1(output)
    # 	output = self.dense2(output)
    # 	return output

    # def compute_mask(self, inputs, mask=None):
    # 	if mask is None:
    # 		return None
    # 	else:
    # 		return tf.math.reduce_all(tf.stack(mask), axis=0)

# def get_data(A):
# 	'''
# 													this function returns edge index and edge attributes sorted in order of occurence from adj matrix A
# 													returns
# 													- edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the graph adjacency (i.e. edges) (i.e. sparse adjacency)
# 					- edge_attr: edge features matrix (sorted by edge apperance in edge_index)
# 	'''
# 	pass


# def from_edge_idx(edge_index, edge_attr, batch):
# 	'''
# 					this function returns A from edge_index and edge_attributes
# 	'''
# 	return pyg_utils.to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr)

# class Reduce(layers.Layer):
# 	def __init__(self, reduction='sum', *args, **kwargs):
# 		super(Reduce, self).__init__(*args, **kwargs)
# 		self.reduction = reduction

# 	def _parse_inputs_and_mask(self, inputs, mask=None):
# 		data, segment_ids, target = inputs

# 		# Handle missing masks
# 		if mask is not None:
# 			data_mask = mask[0]
# 		else:
# 			data_mask = None

# 		return data, segment_ids, target, data_mask

# 	def call(self, inputs, mask=None):
# 		data, segment_ids, target, data_mask = self._parse_inputs_and_mask(
# 			inputs, mask)
# 		num_segments = tf.shape(target, out_type=segment_ids.dtype)[1]
# 		return batched_segment_op(
# 			data,
# 			segment_ids,
# 			num_segments,
# 			data_mask=data_mask,
# 			reduction=self.reduction)

# 	def get_config(self):
# 		return {'reduction': self.reduction}

# class GAT(pyg_nn.MessagePassing):
# 	'''
# 	code taken from https://github.com/zlpure/cs224w/blob/master/hw2-bundle/q4_starter_code/models.py
# 	'''

# 	def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
# 				 dropout=0, bias=True, **kwargs):
# 		super(GAT, self).__init__(aggr='add', **kwargs)

# 		self.in_channels = in_channels
# 		self.out_channels = out_channels
# 		self.heads = num_heads
# 		self.concat = concat
# 		self.dropout = dropout

# 		############################################################################
# 		#  TODO: Your code here!
# 		# Define the layers needed for the forward function.
# 		# Remember that the shape of the output depends the number of heads.
# 		# Our implementation is ~1 line, but don't worry if you deviate from this.

# 		self.lin = nn.Linear(in_channels, self.heads * out_channels)

# 		############################################################################

# 		############################################################################
# 		#  TODO: Your code here!
# 		# The attention mechanism is a single feed-forward neural network parametrized
# 		# by weight vector self.att. Define the nn.Parameter needed for the attention
# 		# mechanism here. Remember to consider number of heads for dimension!
# 		# Our implementation is ~1 line, but don't worry if you deviate from this.

# 		self.att = nn.Parameter(torch.Tensor(1, self.heads, 2 * out_channels))

# 		############################################################################

# 		if bias and concat:
# 			self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
# 		elif bias and not concat:
# 			self.bias = nn.Parameter(torch.Tensor(out_channels))
# 		else:
# 			self.register_parameter('bias', None)

# 		nn.init.xavier_uniform_(self.att)
# 		nn.init.zeros_(self.bias)

# 		############################################################################

# 	def forward(self, x, edge_index, size=None):
# 		############################################################################
# 		#  TODO: Your code here!
# 		# Apply your linear transformation to the node feature matrix before starting
# 		# to propagate messages.
# 		# Our implementation is ~1 line, but don't worry if you deviate from this.

# 		x = self.lin(x)
# 		############################################################################

# 		# Start propagating messages.
# 		return self.propagate(edge_index, size=size, x=x)

# 	def message(self, edge_index_i, x_i, x_j, size_i):
# 		#  Constructs messages to node i for each edge (j, i).

# 		############################################################################
# 		#  TODO: Your code here! Compute the attention coefficients alpha as described
# 		# in equation (7). Remember to be careful of the number of heads with
# 		# dimension!
# 		# Our implementation is ~5 lines, but don't worry if you deviate from this.

# 		x_i, x_j = x_i.view(-1, self.heads, self.out_channels), x_j.view(-1,
# 																		 self.heads, self.out_channels)
# 		x_concat = torch.cat((x_i, x_j), dim=-1)
# 		alpha_out = (x_concat * self.att).sum(dim=-1)
# 		alpha_out = F.leaky_relu(alpha_out, negative_slope=0.2)

# 		alpha = pyg_utils.softmax(alpha_out, edge_index_i, size_i)

# 		############################################################################

# 		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

# 		return x_j * alpha.view(-1, self.heads, 1)

# 	def update(self, aggr_out):
# 		# Updates node embedings.
# 		if self.concat is True:
# 			aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
# 		else:
# 			aggr_out = aggr_out.mean(dim=1)

# 		if self.bias is not None:
# 			aggr_out = aggr_out + self.bias
# 		return aggr_out
