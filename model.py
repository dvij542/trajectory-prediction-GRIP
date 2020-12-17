import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, EncoderRNN
#from visualization.wandb_utils import init_wandb, save_model_wandb, log_losses, log_metrics, log_summary 
import numpy as np 

class Model(nn.Module):
	def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
		super().__init__()
		self.decoder_size = 64
		# load graph
		self.graph = Graph(**graph_args)
		# Initialise Adjacency Matrix
		A = np.ones((8, graph_args['num_node'], graph_args['num_node']))
		#self.enc_lstm = torch.nn.LSTM(in_channels,self.encoder_size,1)
		# build networks
		spatial_kernel_size = np.shape(A)[0]
		temporal_kernel_size = 8 #9 #5 # 3
		kernel_size = (temporal_kernel_size, spatial_kernel_size)
		self.dyn_emb = torch.nn.Linear(64,64)
		self.def_dyn_emb = torch.nn.Linear(64,64)
		self.leaky_relu = torch.nn.LeakyReLU(0.1)
		self.relu = torch.nn.ReLU()
		self.ip_emb = nn.Conv2d(
				in_channels,
				64,
				(1, 1),
				(1, 1),
				(0,0),
			)
		
		#self.dec_lstm = torch.nn.LSTM(64, self.decoder_size)
		#self.op = torch.nn.Linear(self.decoder_size,2)
		# best
		self.st_gcn_networks = nn.ModuleList((
			#nn.BatchNorm2d(in_channels),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			#Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			#Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		))

		# initialize parameters for edge importance weighting
		# if edge_importance_weighting:
		# 	self.edge_importance = nn.ParameterList(
		# 		[nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
		# 		)
		# else:
		# 	self.edge_importance = [1] * len(self.st_gcn_networks)

		self.num_node = num_node = 400
		self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
		self.seq2seq = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		self.sigmoid = nn.Sigmoid()
		self.maxpool = nn.MaxPool1d(4)
		self.extract_features = nn.Linear(16,128)
		# self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		# self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)


	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) 
		return now_feat
	
	def reshape_for_dec_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, V = feature.size() 
		#now_feat = feature.permute(0, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = feature.contiguous()
		now_feat = now_feat.view(N*V, C) 
		return now_feat

	def reshape_from_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		#predicted = predicted.permute(1,0,2)
		now_feat = predicted.view(-1, self.num_node, T, 2) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def reshape_from_enc_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, 64) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		x = pra_x
		# input is NCTV
		# forward
		# graph_conv_feature.shape = (N*V,T,C)
		x = self.leaky_relu(self.ip_emb(x)) # (N,64,T,V)
		graph_conv_feature = self.reshape_for_lstm(x)
		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]
		hist_enc1 = self.seq2seq(in_data=graph_conv_feature, encdec=True, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location) # (N*V,64)
		#print(hist_enc.shape)
		
		# Prior probability
		hist_enc = self.sigmoid(self.dyn_emb(hist_enc1.view(hist_enc1.shape[1],hist_enc1.shape[2])))
		hist_enc1 = self.def_dyn_emb(hist_enc1.view(hist_enc1.shape[1],hist_enc1.shape[2]))
		hist_enc = self.reshape_from_enc_lstm(hist_enc) # (N,64,V)
		hist_enco = self.reshape_from_enc_lstm(hist_enc1) # (N,64,V)
		
		for gcn in self.st_gcn_networks:
			hist_enc, _ = gcn(hist_enc, hist_enco, pra_A)
		
		# x has the probability distribution of each planned traj feature
		# print(x.shape)
		x = hist_enc.permute(0,2,1)
		x = self.maxpool(x)
		print(x.shape)
		x = x/(torch.sum(x, axis=1, keepdims=True))
		x = self.extract_features(x)
		x = x.permute(0,2,1)
		
		# prepare for seq2seq lstm model

		#if pra_teacher_forcing_ratio>0 and type(pra_teacher_location) is not type(None):
		#	pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

		# now_predict.shape = (N, T, V*C)
		# now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# # Return back the result in (N,5,T,V) form
		# now_predict_car = self.reshape_from_lstm(now_predict_car) # (N, C, T, V)

		# now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)

		# now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)
		
		x = self.reshape_for_dec_lstm(x) # (n*v,c)
		# x = torch.cat((x,hist_enc1),axis=1)
		
		x = torch.stack([x])
		now_predict = self.seq2seq(in_data=x, encdec=False, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		
		now_predict = self.reshape_from_lstm(now_predict)

		return now_predict 

if __name__ == '__main__':
	model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
	print(model)

