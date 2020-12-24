import sys
sys.path.append('/usr/local/envs/sc-glstm/lib/python3.7/site-packages')
import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle
import itertools
import math

# Please change this to your location
data_root = '/content/drive/MyDrive/trajectory-prediction-GRIP-current_approach_updated/data/'


history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 400 # maximum number of observed objects is 70
neighbor_distance = 20 # meter

# NGSIM data format:
# frame_id, object_id, object_type, position_x, position_y, object_length, pbject_width
total_feature_dimension = 8 + 1 # we add mark "1" to the end of each row to indicate that this row exists

# after zero centralize data max(x)=127.1, max(y)=106.1, thus choose 130

def get_frame_instance_dict(pra_file_path):
	'''
	Read raw data from files and return a dictionary: 
		{frame_id: 
			{object_id: 
				# 8 features
				[frame_id, object_id, object_type, position_x, position_y, object_length, pbject_width, lane]
			}
		}
	'''
	with open(pra_file_path, 'rb') as reader:
		# print(train_file_path)
		content = pickle.load(reader)
		now_dict = {}
		for row in content:
			n_dict = now_dict.get(row[0], {})
			n_dict[row[1]] = row#[2:]
			now_dict[row[0]] = n_dict
	return now_dict

def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
	"""Used for processing the data
	Inputs :-
		pra_now_dict <numpy array> : Frame wise dictionary of size (no of frames, no of vehicles, feature size)
		pra_start_ind <int> : Start frame index
		pra_end_ind <int> : End frame index
		pra_observed_last <int> : Current frame index
	Outputs :-
		reverse angle matrix, dataset, [masks], mean xy
	"""
	visible_object_id_list = list(pra_now_dict[pra_observed_last].keys()) # object_id appears at the last observed frame 
	num_visible_object = len(visible_object_id_list) # number of current observed objects
	#print("yes")
	# compute the mean values of x and y for zero-centralization. 
	visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))
	xy = visible_object_value[:, 3:5].astype(float)
	veh_class = visible_object_value[:,2]
	lane = visible_object_value[:,7]
	y_corr = visible_object_value[:,4]
	mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
	m_xy = np.mean(xy, axis=0)
	mean_xy[3:5] = m_xy

	# compute distance between any pair of two objects
	dist_xy = spatial.distance.cdist(xy, xy)
	dist_mask = np.zeros((max_num_object, max_num_object))
	dist_mask[:num_visible_object, :num_visible_object] = (dist_xy<neighbor_distance).astype(float)
	# compute relative x and y position matrix
	
	rely = np.zeros((max_num_object,max_num_object))
	relx = np.zeros((max_num_object,max_num_object))
	for i,j in itertools.product(range(xy.shape[0]),range(xy.shape[0])) :
		relx[i,j] = xy[i,0] - xy[j,0]
		rely[i,j] = xy[i,1] - xy[j,1]
	
	relxy = np.array((relx,rely))
	#change to radians
	#print(heading)

	#now_feature[3:5, :, :] = xy
	# assign person class binary matrix
	classi = np.zeros((max_num_object,max_num_object))
	classj = np.zeros((max_num_object,max_num_object))
	lanei = np.zeros((max_num_object,max_num_object))
	lanej = np.zeros((max_num_object,max_num_object))
	fronti = np.zeros((max_num_object,max_num_object))
	frontj = np.zeros((max_num_object,max_num_object))
	identity_matrix = np.zeros((num_visible_object, num_visible_object))
	for i,j in itertools.product(range(xy.shape[0]),range(xy.shape[0])) :
		classi[i,j] = (veh_class[i]==3).astype(float)
		classj[i,j] = (veh_class[j]==3).astype(float)
		lanei[i,j] = ((lane[i]-lane[j])==1).astype(float)
		lanej[i,j] = ((lane[j]-lane[i])==1).astype(float)
		fronti[i,j] = ((y_corr[i]-y_corr[j])>0 and ((lane[i]-lane[j])==0)).astype(float)
		frontj[i,j] = ((y_corr[j]-y_corr[i])>0 and ((lane[j]-lane[i])==0)).astype(float)
		identity_matrix[i,j] = (i==j)
	# Store the distances in a fixed size matrix (neighbour_matrix)
	neighbor_matrix = np.zeros((max_num_object, max_num_object))
	#identity_matrix = np.zeros((num_visible_object, num_visible_object))
	neighbor_matrix[:num_visible_object, :num_visible_object] = (identity_matrix)
	
	now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind) for val in pra_now_dict[x].keys()])
	non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
	num_non_visible_object = len(non_visible_object_id_list)
	print('No of objects :', num_visible_object+num_non_visible_object)
	# for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
	object_feature_list = []
	# non_visible_object_feature_list = []
	print(pra_start_ind, pra_end_ind)
	for frame_ind in range(pra_start_ind, pra_end_ind,5):	
		# we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
		# -mean_xy is used to zero_centralize data
		# now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
		print(frame_ind)
		now_frame_feature_dict = {obj_id : (list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] if obj_id in visible_object_id_list else list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[0]) for obj_id in pra_now_dict[frame_ind] }
		# if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
		now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
		object_feature_list.append(now_frame_feature)

	# object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
	
	print((object_feature_list))
	object_feature_list = np.array(object_feature_list)
	
	# object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
	object_frame_feature = np.zeros((max_num_object, (pra_end_ind-pra_start_ind)//5, total_feature_dimension))
	
	# np.transpose(object_feature_list, (1,0,2))
	object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))
	#if (object_frame_feature[:num_visible_object,5,3]-object_frame_feature[:num_visible_object,4,3]) != 0 :
	vel_angle = np.arctan(np.abs(object_frame_feature[:num_visible_object,5,4]-object_frame_feature[:num_visible_object,4,4])/np.maximum(np.abs(object_frame_feature[:num_visible_object,5,3]-object_frame_feature[:num_visible_object,4,3]),0.00001))
	vel_angle = (np.sign(object_frame_feature[:num_visible_object,5,4]-object_frame_feature[:num_visible_object,4,4])==1)*(np.sign(object_frame_feature[:num_visible_object,5,3]-object_frame_feature[:num_visible_object,4,3])==1)*vel_angle + (np.sign(object_frame_feature[:num_visible_object,5,4]-object_frame_feature[:num_visible_object,4,4])==-1)*(np.sign(object_frame_feature[:num_visible_object,5,3]-object_frame_feature[:num_visible_object,4,3])==1)*(-vel_angle) + (np.sign(object_frame_feature[:num_visible_object,5,4]-object_frame_feature[:num_visible_object,4,4])==1)*(np.sign(object_frame_feature[:num_visible_object,5,3]-object_frame_feature[:num_visible_object,4,3])==-1)*(math.pi-vel_angle) + (np.sign(object_frame_feature[:num_visible_object,5,4]-object_frame_feature[:num_visible_object,4,4])==-1)*(np.sign(object_frame_feature[:num_visible_object,5,3]-object_frame_feature[:num_visible_object,4,3])==-1)*(math.pi+vel_angle)
	vel_angle = vel_angle*(object_frame_feature[:num_visible_object,4,3]!=0)
	#if (object_frame_feature[:num_visible_object,5,4]-object_frame_feature[:num_visible_object,4,4])<0 && 
	if np.isnan(vel_angle).sum(axis=0) == 1 :
		print("yes")
	angle = np.zeros(max_num_object)
	sin_angle = np.sin(angle)
	cos_angle = np.cos(angle)
	angleij = np.zeros((max_num_object,max_num_object))
	for i,j in itertools.product(range(xy.shape[0]),range(xy.shape[0])) :
		angleij[i,j] = vel_angle[i]-vel_angle[j]
		if angleij[i,j]<-math.pi :
			angleij[i,j] += 2*math.pi
		if angleij[i,j]>math.pi :
			angleij[i,j] -= 2*math.pi
	angle_mat = np.array(
			[[cos_angle, sin_angle],
			[-sin_angle, cos_angle]])
	rev_angle_mat = np.array(
			[[cos_angle, -sin_angle],
			[sin_angle, cos_angle]])
	
	object_frame_feature[:,:,3:5] = np.einsum('abv,vtb->vta',angle_mat,object_frame_feature[:,:,3:5])
	out_relxy = np.einsum('abw,bvw->avw', angle_mat, relxy)
	
	relx = out_relxy[0]
	rely = out_relxy[1]
	new_mask = (object_frame_feature[:, 1:, 3:4]!=0) * (object_frame_feature[:, :-1, 3:4]!=0) 
	# data contains velocity
	data = np.zeros((max_num_object, (pra_end_ind-pra_start_ind)//5, 2))
	data[:, 1:,:2] = (object_frame_feature[:, 1:,3:5] - object_frame_feature[:, :-1, 3:5]).astype(float) * new_mask.astype(float)
	data[:, 0,:2] = 0	
	total_avg_vel = data.sum(axis=0).sum(axis=-2)/new_mask.sum(axis=0).sum(axis=-2)
	#print(total_avg_vel)
	#print(object_frame_feature[0,history_frames-1,3:5]-object_frame_feature[0,history_frames-2,3:5])
	return rev_angle_mat, object_frame_feature, np.array((neighbor_matrix,fronti,frontj,lanei,lanej,dist_mask)), m_xy
	

def generate_train_data(pra_file_path):
	'''
	Read data from $pra_file_path, and split data into clips with $total_frames length. 
	Return: feature and adjacency_matrix
		feture: (N, C, T, V) 
			N is the number of training data 
			C is the dimension of features, 7 raw_feature + 1 mark (valid data or not)
			T is the temporal length of the data. history_frames + future_frames
			V is the maximum number of objects. zero-padding for less objects. 
	'''
	now_dict = get_frame_instance_dict(pra_file_path)
	frame_id_set = sorted(set(now_dict.keys()))
	frame_id_set = frame_id_set[::5]
	all_feature_list = []
	all_adjacency_list = []
	all_mean_list = []
	rev_angle_mat_list = []
	for start_ind in frame_id_set[:2000:total_frames]:
		print(start_ind, '/', len(frame_id_set)*5)
		start_ind = int(start_ind)
		end_ind = int(start_ind + 5*total_frames)
		observed_last = start_ind + 5*(history_frames - 1)
		rev_angle_mat, object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

		all_feature_list.append(object_frame_feature)
		all_adjacency_list.append(neighbor_matrix)	
		all_mean_list.append(mean_xy)	
		rev_angle_mat_list.append(rev_angle_mat)
	# (N, V, T, C) --> (N, C, T, V)
	all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
	rev_angle_mat_list = np.array(rev_angle_mat_list)
	
	all_adjacency_list = np.array(all_adjacency_list)
	all_mean_list = np.array(all_mean_list)
	# print(all_feature_list.shape, all_adjacency_list.shape)
	return rev_angle_mat_list, all_feature_list, all_adjacency_list, all_mean_list


def generate_test_data(pra_file_path):
	'''
	Read data from $pra_file_path, and split data into clips with $total_frames length. 
	Return: feature and adjacency_matrix
		feture: (N, C, T, V) 
			N is the number of training data 
			C is the dimension of features, 7 raw_feature + 1 mark (valid data or not)
			T is the temporal length of the data. history_frames + future_frames
			V is the maximum number of objects. zero-padding for less objects. 
	'''
	now_dict = get_frame_instance_dict(pra_file_path)
	frame_id_set = sorted(set(now_dict.keys()))
	
	all_feature_list = []
	all_adjacency_list = []
	all_mean_list = []
	rev_angle_mat_list = []
	# get all start frame

	start_frame_id_list = frame_id_set[::history_frames]
	print(start_frame_id_list)
	for start_ind in start_frame_id_list:
		start_ind = int(start_ind)
		end_ind = int(start_ind + history_frames)
		observed_last = start_ind + history_frames - 1
		# print(start_ind, end_ind)
		rev_angle_mat, object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

		all_feature_list.append(object_frame_feature)
		all_adjacency_list.append(neighbor_matrix)	
		all_mean_list.append(mean_xy)
		rev_angle_mat_list.append(rev_angle_mat)

	# (N, V, T, C) --> (N, C, T, V)
	all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
	all_adjacency_list = np.array(all_adjacency_list)
	all_mean_list = np.array(all_mean_list)
	rev_angle_mat_list = np.array(rev_angle_mat_list)
	# print(all_feature_list.shape, all_adjacency_list.shape)
	return rev_angle_mat_list, all_feature_list, all_adjacency_list, all_mean_list


def generate_data(pra_file_path_list, pra_is_train=True):
   
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    all_rev_angle_mat = []
    for file_path in pra_file_path_list:
	    print(file_path)
	    if pra_is_train:
		    rev_angle_mat, now_data, now_adjacency, now_mean_xy = generate_train_data(file_path)
	    else:
		    rev_angle_mat, now_data, now_adjacency, now_mean_xy = generate_test_data(file_path)
	    all_data.extend(now_data)
	    all_adjacency.extend(now_adjacency)
	    all_mean_xy.extend(now_mean_xy)
	    all_rev_angle_mat.extend(rev_angle_mat)

    all_data = np.array(all_data) #(N, C, T, V)=(5010, 11, 12, 70) Train
    all_adjacency = np.array(all_adjacency) #(5010, 70, 70) Train
    all_mean_xy = np.array(all_mean_xy) #(5010, 2) Train
    all_rev_angle_mat = np.array(all_rev_angle_mat)
	# Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
	# Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)
    print(np.shape(all_rev_angle_mat),np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))
    x = [all_rev_angle_mat, all_data, all_adjacency, all_mean_xy]
	# save training_data and trainjing_adjacency into a file.
    if pra_is_train:
	    save_path = '/content/drive/MyDrive/trajectory-prediction-GRIP-current_approach_updated/train_data.pkl'
    else:
	    save_path = 'test_data.pkl'
    with open(save_path, 'wb') as writer:
	    pickle.dump(x, writer)
		

if __name__ == '__main__':
	train_file_path_list = sorted(glob.glob(os.path.join(data_root, '*.pkl')))
	#test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

	print('Generating Training Data.')
	generate_data(train_file_path_list, pra_is_train=True)
	
	#print('Generating Testing Data.')
	#generate_data(test_file_path_list, pra_is_train=False)

