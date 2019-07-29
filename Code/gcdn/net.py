import numpy as np
import tensorflow as tf
import time
import random

from tensorflow.python.client import timeline


class NET:


	def __init__(self, config):

		self.config = config 
		self.N = config.N

		######### not running out gpu sources ##########	
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True	
		tf_config.allow_soft_placement = True
		self.sess = tf.Session(config=tf_config)

		################## profiling ###################
		#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#self.run_metadata = tf.RunMetadata()

		############### define variables ###############

		with tf.device('/cpu:0'):
		
			self.W = {}
			self.b = {}
			self.scale={}
			self.beta={}
			self.pop_mean={}
			self.pop_var={}
			self.alpha={}
			self.beta={}

			self.dn_vars = []

			# pre
			name_block = "pre"
			self.W[name_block+"3_l_0"] = tf.get_variable(name_block+"3_l_0", [3, 3, config.patch_size[2], config.pre_Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
			#self.create_bn_variables(name_block+"3_0", config.pre_Nfeat)
			self.W[name_block+"5_l_0"] = tf.get_variable(name_block+"5_l_0", [5, 5, config.patch_size[2], config.pre_Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
			#self.create_bn_variables(name_block+"5_0", config.pre_Nfeat)
			self.W[name_block+"7_l_0"] = tf.get_variable(name_block+"7_l_0", [7, 7, config.patch_size[2], config.pre_Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
			#self.create_bn_variables(name_block+"7_0", config.pre_Nfeat)
			self.dn_vars = self.dn_vars + [self.W[name_block+"3_l_0"],self.W[name_block+"5_l_0"],self.W[name_block+"7_l_0"]]
			for i in range(1,config.pre_n_layers):
				self.W[name_block+"3_l_" + str(i)] = tf.get_variable(name_block+"3_l_" + str(i), [3, 3, config.pre_Nfeat, config.pre_Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
				#self.create_bn_variables(name_block+"3_"+str(i), config.pre_Nfeat)
				self.W[name_block+"5_l_" + str(i)] = tf.get_variable(name_block+"5_l_" + str(i), [5, 5, config.pre_Nfeat, config.pre_Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
				#self.create_bn_variables(name_block+"5_"+str(i), config.pre_Nfeat)
				self.W[name_block+"7_l_" + str(i)] = tf.get_variable(name_block+"7_l_" + str(i), [7, 7, config.pre_Nfeat, config.pre_Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
				#self.create_bn_variables(name_block+"7_"+str(i), config.pre_Nfeat)
				self.dn_vars = self.dn_vars + [self.W[name_block+"3_l_"+str(i)],self.W[name_block+"5_l_"+str(i)],self.W[name_block+"7_l_"+str(i)]]

			# pregconv
			name_block = "pregconv"
			for i in range(config.pregconv_n_layers):
				self.create_gconv_variables(name_block+"3", i, config.pre_Nfeat, config.pre_fnet_Nfeat, config.pre_Nfeat, config.rank_theta, config.stride_pregconv, config.stride_pregconv)
				self.create_gconv_variables(name_block+"5", i, config.pre_Nfeat, config.pre_fnet_Nfeat, config.pre_Nfeat, config.rank_theta, config.stride_pregconv, config.stride_pregconv)
				self.create_gconv_variables(name_block+"7", i, config.pre_Nfeat, config.pre_fnet_Nfeat, config.pre_Nfeat, config.rank_theta, config.stride_pregconv, config.stride_pregconv)
			#self.create_bn_variables(name_block, config.Nfeat)

			# hpf
			name_block = "hpf"
			self.create_conv_variables(name_block, 0, config.Nfeat, config.Nfeat)
			self.create_bn_variables(name_block+"_c_"+"_"+str(0), config.Nfeat)
			for i in range(config.hpf_n_layers):
				self.create_gconv_variables(name_block, i, config.Nfeat, config.hpf_fnet_Nfeat, config.Nfeat, config.rank_theta, config.stride, config.stride)	
				#self.create_bn_variables(name_block+"_"+str(i), config.Nfeat)	

			# prox
			name_block = "prox"
			for i in range(config.prox_n_layers):
				self.create_conv_variables(name_block, i, config.Nfeat, config.Nfeat)
				self.create_bn_variables(name_block+"_c_"+"_"+str(i), config.Nfeat)
				for j in range(config.lpf_n_layers):
					self.create_gconv_variables(name_block+str(i), j, config.Nfeat, config.prox_fnet_Nfeat, config.Nfeat, config.rank_theta, config.stride, config.stride)
					self.create_bn_variables(name_block+str(i)+"_"+str(j), config.Nfeat)
				self.alpha["alpha_"+str(i)] = tf.get_variable("alpha_"+str(i), [], dtype=tf.float32, initializer=tf.constant_initializer(0.5)) 
				self.beta["beta_"+str(i)] = tf.get_variable("beta_"+str(i), [], dtype=tf.float32, initializer=tf.constant_initializer(0.5)) 
				self.dn_vars = self.dn_vars + [self.alpha["alpha_"+str(i)], self.beta["beta_"+str(i)]]

			# last
			name_block = "last"
			self.create_gconv_variables(name_block, 0, config.Nfeat, config.prox_fnet_Nfeat, config.patch_size[2], config.rank_theta, config.stride, config.patch_size[2])


			# only optimize these vars
			# self.opt_vars=[]
			# for i in range(2,self.config.prox_n_layers):
			# 	self.opt_vars = self.opt_vars + [self.alpha["alpha_"+str(i)]]
			# 	self.opt_vars = self.opt_vars + [self.beta["beta_"+str(i)]]
			# 	for j in range(self.config.lpf_n_layers):
			# 		name = "prox"+str(i) + "_nl_" + str(j) + "_flayer0"
			# 		self.opt_vars = self.opt_vars + [self.W[name],self.b[name]]
			# 		name = "prox"+str(i) + "_nl_" + str(j) + "_flayer1"
			# 		self.opt_vars = self.opt_vars + [self.W[name+"_th1"],self.b[name+"_th1"],self.W[name+"_th2"],self.b[name+"_th2"],self.W[name+"_thl"],self.b[name+"_thl"]]	
			# 		name = "prox"+str(i) + "_l_" + str(j)
			# 		self.opt_vars = self.opt_vars + [self.W[name]]
			# 		name = "prox"+str(i) + "_" + str(j)
			# 		self.opt_vars = self.opt_vars + [self.b[name]]
			# 		self.opt_vars = self.opt_vars + [self.scale['bn_scale_'+name], self.beta['bn_beta_'+name], self.pop_mean['bn_pop_mean_'+name], self.pop_var['bn_pop_var_'+name ]]
			# name = "last" + "_nl_" + str(0) + "_flayer0"
			# self.opt_vars = self.opt_vars + [self.W[name],self.b[name]]
			# name = "last" + "_nl_" + str(0) + "_flayer1"
			# self.opt_vars = self.opt_vars + [self.W[name+"_th1"],self.b[name+"_th1"],self.W[name+"_th2"],self.b[name+"_th2"],self.W[name+"_thl"],self.b[name+"_thl"]]	
			# name = "last" + "_l_" + str(0)
			# self.opt_vars = self.opt_vars + [self.W[name]]
			# name = "last" + "_" + str(0)
			# self.opt_vars = self.opt_vars + [self.b[name]]
			

			############ define placeholders ##############
			self.x_clean = tf.placeholder("float", [None, config.patch_size[0], config.patch_size[1], config.patch_size[2]], name="clean_image")
			self.x_noisy = tf.placeholder("float", [None, config.patch_size[0], config.patch_size[1], config.patch_size[2]], name="noisy_image")
			self.is_training = tf.placeholder(tf.bool, (), name="is_training")
			self.local_mask = tf.placeholder("float", [1, config.N, config.N], name="local_mask")

			self.xs_clean = tf.split(self.x_clean, self.config.Ngpus, axis=0)
			self.xs_noisy = tf.split(self.x_noisy, self.config.Ngpus, axis=0)

			############# computational graph #############
			self.make_computational_graph()

			################# summaries ###################
			#tf.summary.scalar('loss', self.loss)
			#tf.summary.scalar('PSNR', self.psnr)
			#tf.summary.scalar('alpha_0', self.alpha["alpha_0"])
			#tf.summary.image('denoised_image',tf.expand_dims(self.x_hat[0,:,:,:],0))
			#tf.summary.image('noisy_image',tf.expand_dims(self.x_noisy[0,:,:,:],0))
			#tf.summary.image('clean_image',tf.expand_dims(self.x_clean[0,:,:,:],0))
			#self.summaries = tf.summary.merge_all()
			#self.train_summaries_writer = tf.summary.FileWriter(self.config.log_dir+'train/', self.sess.graph)
			#self.val_summaries_writer = tf.summary.FileWriter(self.config.log_dir+'val/', self.sess.graph)



	def average_gradients(self, tower_grads):

		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			# Note that each grad_and_vars looks like the following:
			#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
			grads = []
			for g, v in grad_and_vars:
				# Add 0 dimension to the gradients to represent the tower.
				expanded_g = tf.expand_dims(g, 0)
				#if g is not None:
				#	expanded_g = tf.expand_dims(g, 0)
				#else:
				#	expanded_g = tf.expand_dims(tf.zeros_like(v), 0)
				# Append on a 'tower' dimension which we will average over below.
				grads.append(expanded_g)

			# Average over the 'tower' dimension.
			grad = tf.concat(axis=0, values=grads)
			grad = tf.reduce_mean(grad, 0)

			# Keep in mind that the Variables are redundant because they are shared
			# across towers. So .. we will just return the first tower's pointer to
			# the Variable.
			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
		
		return average_grads


	def create_gconv_variables(self, name_block, i, in_feat, fnet_feat, out_feat, rank_theta, stride_th1, stride_th2):
	
		name = name_block + "_nl_" + str(i) + "_flayer0"
		self.W[name] = tf.get_variable(name, [in_feat, fnet_feat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
		self.b[name] = tf.get_variable("b_"+name, [1, fnet_feat], dtype=tf.float32, initializer=tf.zeros_initializer()) 	
		self.dn_vars = self.dn_vars + [self.W[name], self.b[name]]
		name = name_block + "_nl_" + str(i) + "_flayer1"
		self.W[name+"_th1"] = tf.get_variable(name+"_th1", [fnet_feat, stride_th1*rank_theta], dtype=tf.float32, initializer=tf.random_normal_initializer(0,1.0/(np.sqrt(fnet_feat+0.0)*np.sqrt(in_feat+0.0))))
		self.b[name+"_th1"] = tf.get_variable(name+"_b_th1", [1, rank_theta, in_feat], dtype=tf.float32, initializer=tf.zeros_initializer())
		self.W[name+"_th2"] = tf.get_variable(name+"_th2", [fnet_feat, stride_th2*rank_theta], dtype=tf.float32, initializer=tf.random_normal_initializer(0,1.0/(np.sqrt(fnet_feat+0.0)*np.sqrt(in_feat+0.0))))
		self.b[name+"_th2"] = tf.get_variable(name+"_b_th2", [1, rank_theta, out_feat], dtype=tf.float32, initializer=tf.zeros_initializer())
		self.W[name+"_thl"] = tf.get_variable(name+"_thl", [fnet_feat, rank_theta], dtype=tf.float32, initializer=tf.random_normal_initializer(0,1.0/np.sqrt(rank_theta+0.0)))
		self.b[name+"_thl"] = tf.get_variable(name+"_b_thl", [1, rank_theta], dtype=tf.float32, initializer=tf.zeros_initializer())
		self.dn_vars = self.dn_vars + [self.W[name+"_th1"],self.b[name+"_th1"],self.W[name+"_th2"],self.b[name+"_th2"],self.W[name+"_thl"],self.b[name+"_thl"]]	
		name = name_block + "_l_" + str(i)
		self.W[name] = tf.get_variable(name, [3, 3, in_feat, out_feat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
		self.dn_vars = self.dn_vars + [self.W[name]]
		name = name_block + "_" + str(i)
		self.b[name] = tf.get_variable(name, [1, out_feat], dtype=tf.float32, initializer=tf.zeros_initializer()) 
		self.dn_vars = self.dn_vars + [self.b[name]]


	def create_conv_variables(self, name_block, i, in_feat, out_feat):
	
		name = name_block + "_c_" + str(i)
		self.W[name] = tf.get_variable(name, [3, 3, in_feat, out_feat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
		self.dn_vars = self.dn_vars + [self.W[name]]
		name = name_block + "_cb_" + str(i)
		self.b[name] = tf.get_variable(name, [1, out_feat], dtype=tf.float32, initializer=tf.zeros_initializer()) 
		self.dn_vars = self.dn_vars + [self.b[name]]


	def create_bn_variables(self, name, Nfeat):

		self.scale['bn_scale_'+name] = tf.get_variable('bn_scale_'+name, [Nfeat], initializer=tf.ones_initializer())
		self.beta['bn_beta_'+name]  = tf.get_variable('bn_beta_'+name , [Nfeat], initializer=tf.constant_initializer(0.0))
		self.pop_mean['bn_pop_mean_'+name] = tf.get_variable('bn_pop_mean_'+name, [Nfeat], initializer=tf.constant_initializer(0.0), trainable=False)
		self.pop_var['bn_pop_var_'+name ]  = tf.get_variable('bn_pop_var_'+name , [Nfeat], initializer=tf.ones_initializer(), trainable=False)
		self.dn_vars = self.dn_vars + [self.scale['bn_scale_'+name], self.beta['bn_beta_'+name]]


	def compute_graph(self, h):
		
		id_mat = 2*tf.eye(self.N)

		h = tf.cast(h, tf.float64)

		sq_norms = tf.reduce_sum(h*h,2) # (B,N)
		D = tf.abs( tf.expand_dims(sq_norms, 2) + tf.expand_dims(sq_norms, 1) - 2*tf.matmul(h, h, transpose_b=True) ) # (B, N, N)
		D = tf.cast(D, tf.float32)	
		D = tf.multiply(D, self.local_mask)
		D = D - id_mat

		h = tf.cast(h, tf.float32) 

		return D


	# same as new tf.roll but only for 3D input and axis=2
	def myroll(self, h, shift=0, axis=2):
			
		h_len = h.get_shape()[2]
		return tf.concat([h[:,:,h_len-shift:], h[:,:,:h_len-shift]], axis=2)


	def gconv(self, h, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False, D=[]):

		if compute_graph:
			D = self.compute_graph(h)

		_, top_idx = tf.nn.top_k(-D, self.config.min_nn+1) # (B, N, d+1)
		top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:,:,0],2), [1, 1, self.config.min_nn-8]), [-1, self.N*(self.config.min_nn-8)]) # (B, N*d)
		top_idx = tf.reshape(top_idx[:,:,9:],[-1, self.N*(self.config.min_nn-8)]) # (B, N*d)

		x_tilde1 = tf.batch_gather(h, top_idx) # (B, K, dlm1)		
		x_tilde2 = tf.batch_gather(h, top_idx2) # (B, K, dlm1)
		labels = x_tilde1 - x_tilde2 # (B, K, dlm1)
		x_tilde1 = tf.reshape(x_tilde1, [-1, in_feat]) # (B*K, dlm1)
		labels = tf.reshape(labels, [-1, in_feat]) # (B*K, dlm1)
		d_labels = tf.reshape( tf.reduce_sum(labels*labels, 1), [-1, self.config.min_nn-8]) # (B*N, d)

		name_flayer = name + "_flayer0"
		labels = tf.nn.leaky_relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer]) #  (B*K, F)
		name_flayer = name + "_flayer1"
		labels_exp = tf.expand_dims(labels, 1) # (B*K, 1, F)
		labels1 = labels_exp+0.0
		for ss in range(1, in_feat/stride_th1):
			labels1 = tf.concat( [labels1, self.myroll(labels_exp, shift=(ss+1)*stride_th1, axis=2)], axis=1 ) # (B*K, dlm1/stride, dlm1)
		labels2 = labels_exp+0.0
		for ss in range(1, out_feat/stride_th2):
			labels2 = tf.concat( [labels2, self.myroll(labels_exp, shift=(ss+1)*stride_th2, axis=2)], axis=1 ) # (B*K, dl/stride, dlm1)
		theta1 = tf.matmul( tf.reshape(labels1, [-1, in_feat]), self.W[name_flayer+"_th1"] )  # (B*K*dlm1/stride, R*stride)
		theta1 = tf.reshape(theta1, [-1, self.config.rank_theta, in_feat] ) + self.b[name_flayer+"_th1"]
		theta2 = tf.matmul( tf.reshape(labels2, [-1, in_feat]), self.W[name_flayer+"_th2"] )  # (B*K*dl/stride, R*stride)
		theta2 = tf.reshape(theta2, [-1, self.config.rank_theta,  out_feat] ) + self.b[name_flayer+"_th2"]	
		thetal = tf.expand_dims( tf.matmul(labels, self.W[name_flayer+"_thl"]) + self.b[name_flayer+"_thl"], 2 ) # (B*K, R, 1)

		x = tf.matmul(theta1, tf.expand_dims(x_tilde1,2)) # (B*K, R, 1)
		x = tf.multiply(x, thetal) # (B*K, R, 1)
		x = tf.matmul(theta2, x, transpose_a=True)[:,:,0] # (B*K, dl)

		x = tf.reshape(x, [-1, self.config.min_nn-8, out_feat]) # (N, d, dl)
		x = tf.multiply(x, tf.expand_dims(tf.exp(-tf.div(d_labels,10)),2)) # (N, d, dl)
		x = tf.reduce_mean(x, 1) # (N, dl)
		x = tf.reshape(x,[-1, self.N, out_feat]) # (B, N, dl)
		
		if return_graph:
			return x, D
		else:
			return x


	def lnl_aggregation(self, h_l, h_nl, b):
		
		return tf.div(h_l + h_nl, 2) + b
		#return h_l + b


	def batch_norm_wrapper(self, inputs, name, decay = 0.999):
		
		def bn_train():
			if len(inputs.get_shape())==4:
				# for convolutional activations of size (batch, height, width, depth)
				batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
			if len(inputs.get_shape())==3:
				# for activations of size (batch, points, features)
				batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
			if len(inputs.get_shape())==2:
				# for fully connected activations of size (batch, features)
				batch_mean, batch_var = tf.nn.moments(inputs,[0])
			train_mean = tf.assign(self.pop_mean['bn_pop_mean_'+name], self.pop_mean['bn_pop_mean_'+name] * decay + batch_mean * (1 - decay))
			train_var = tf.assign(self.pop_var['bn_pop_var_'+name], self.pop_var['bn_pop_var_'+name] * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.beta['bn_beta_'+name], self.scale['bn_scale_'+name], 1e-3)

		def bn_test():
			return tf.nn.batch_normalization(inputs, self.pop_mean['bn_pop_mean_'+name], self.pop_var['bn_pop_var_'+name], self.beta['bn_beta_'+name], self.scale['bn_scale_'+name], 1e-3)

		normalized = tf.cond( self.is_training, bn_train, bn_test )
		return normalized



	def tower_loss(self, i_gpu):

		def noise_extract(h):

			# pre
			name_block = "pre"
			paddings3 = tf.constant([[0,0],[1, 1],[1, 1],[0, 0]])
			h3 = h + 0.0
			for i in range(self.config.pre_n_layers):
				h3 = tf.nn.conv2d(tf.pad(h3, paddings3, "REFLECT"), self.W[name_block+"3_l_"+str(i)], strides=[1,1,1,1], padding="VALID")
				#h3 = self.batch_norm_wrapper(h3, name_block+"3_"+str(i))
				h3 = tf.nn.leaky_relu(h3)
			h3 = tf.reshape(h3, [-1, self.N, self.config.pre_Nfeat])
			paddings5 = tf.constant([[0,0],[2, 2],[2, 2],[0, 0]])
			h5 = h + 0.0
			for i in range(self.config.pre_n_layers):
				h5 = tf.nn.conv2d(tf.pad(h5, paddings5, "REFLECT"), self.W[name_block+"5_l_"+str(i)], strides=[1,1,1,1], padding="VALID")
				#h5 = self.batch_norm_wrapper(h5, name_block+"5_"+str(i))
				h5 = tf.nn.leaky_relu(h5)
			h5 = tf.reshape(h5, [-1, self.N, self.config.pre_Nfeat])
			paddings7 = tf.constant([[0,0],[3, 3],[3, 3],[0, 0]])
			h7 = h + 0.0
			for i in range(self.config.pre_n_layers):
				h7 = tf.nn.conv2d(tf.pad(h7, paddings7, "REFLECT"), self.W[name_block+"7_l_"+str(i)], strides=[1,1,1,1], padding="VALID")
				#h7 = self.batch_norm_wrapper(h7, name_block+"7_"+str(i))
				h7 = tf.nn.leaky_relu(h7)
			h7 = tf.reshape(h7, [-1, self.N, self.config.pre_Nfeat])

			# pregconv
			name_block = "pregconv"
			for i in range(self.config.pregconv_n_layers):
				h3_nl = self.gconv(h3, name_block+"3_nl_"+str(i), self.config.pre_Nfeat, self.config.pre_Nfeat, self.config.stride_pregconv, self.config.stride_pregconv, compute_graph=True, return_graph=False)
				h3_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h3,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.pre_Nfeat]), paddings3, "REFLECT"), self.W[name_block+"3_l_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.pre_Nfeat])
				h3 = self.lnl_aggregation(h3_l, h3_nl, self.b[name_block+"3_"+str(i)])
				h5_nl = self.gconv(h5, name_block+"5_nl_"+str(i), self.config.pre_Nfeat, self.config.pre_Nfeat, self.config.stride_pregconv, self.config.stride_pregconv, compute_graph=True, return_graph=False)
				h5_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h3,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.pre_Nfeat]), paddings3, "REFLECT"), self.W[name_block+"5_l_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.pre_Nfeat])
				h5 =  self.lnl_aggregation(h5_l, h5_nl, self.b[name_block+"5_"+str(i)])
				h7_nl = self.gconv(h7, name_block+"7_nl_"+str(i), self.config.pre_Nfeat, self.config.pre_Nfeat, self.config.stride_pregconv, self.config.stride_pregconv, compute_graph=True, return_graph=False)
				h7_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h7,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.pre_Nfeat]), paddings3, "REFLECT"), self.W[name_block+"7_l_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.pre_Nfeat])
				h7 =  self.lnl_aggregation(h7_l, h7_nl, self.b[name_block+"7_"+str(i)])
			h = tf.concat([h3, h5, h7], axis=2)
			#h = self.batch_norm_wrapper(h, name_block)
			h = tf.nn.leaky_relu(h)

			# hpf
			name_block = "hpf"
			h_hpf = h + 0.0
			h_hpf = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h_hpf,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+"_c_"+str(0)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.Nfeat]) + self.b[name_block+"_cb_"+str(0)]
			h_hpf = self.batch_norm_wrapper(h_hpf, name_block+"_c_"+"_"+str(0))
			h_hpf = tf.nn.leaky_relu(h_hpf)
			for i in range(self.config.hpf_n_layers):
				if i==0:
					h_hpf_nl, D = self.gconv(h_hpf, name_block+"_nl_"+str(i), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=True, return_graph=True)
				else:		
					h_hpf_nl = self.gconv(h_hpf, name_block+"_nl_"+str(i), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=False, return_graph=False, D=D)
				h_hpf_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h_hpf,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+"_l_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.Nfeat])
				h_hpf = self.lnl_aggregation(h_hpf_l, h_hpf_nl, self.b[name_block+"_"+str(i)])
				#h_hpf = self.batch_norm_wrapper(h_hpf, name_block+"_"+str(i))
				h_hpf = tf.nn.leaky_relu(h_hpf)

			# prox
			name_block = "prox"
			for i in range(self.config.prox_n_layers):
				h = self.beta["beta_"+str(i)]*h_hpf + (1-self.alpha["alpha_"+str(i)])*h
				h_old = h + 0.0
				h = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+"_c_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.Nfeat]) + self.b[name_block+"_cb_"+str(i)]
				h = self.batch_norm_wrapper(h, name_block+"_c_"+"_"+str(i))
				h = tf.nn.leaky_relu(h)
				for j in range(self.config.lpf_n_layers):
					if j==0:
						h_nl, D = self.gconv(h, name_block+str(i)+"_nl_"+str(j), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=True, return_graph=True)
					else:
						h_nl = self.gconv(h, name_block+str(i)+"_nl_"+str(j), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=False, return_graph=False, D=D)		
					h_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+str(i)+"_l_"+str(j)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.Nfeat])	
					h = self.lnl_aggregation(h_l, h_nl, self.b[name_block+str(i)+"_"+str(j)])
					h = self.batch_norm_wrapper(h, name_block+str(i)+"_"+str(j))
					h = tf.nn.leaky_relu(h)
				h = h + h_old

			# last
			name_block = "last"
			h_nl = self.gconv(h, name_block+"_nl_0", self.config.Nfeat, self.config.patch_size[2], self.config.stride, self.config.patch_size[2], compute_graph=True, return_graph=False)
			h_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+"_l_0"], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.patch_size[2]])	
			h = self.lnl_aggregation(h_l, h_nl, self.b[name_block+"_0"])

			h = tf.reshape(h, [-1, self.config.patch_size[0], self.config.patch_size[1], self.config.patch_size[2]])

			return h

		n_hat = noise_extract( self.xs_noisy[i_gpu] ) 
		x_hat = self.xs_noisy[i_gpu] - n_hat

		loss = tf.losses.mean_squared_error(self.xs_noisy[i_gpu]-self.xs_clean[i_gpu], n_hat)
		#loss = tf.losses.absolute_difference(self.xs_noisy[i_gpu]-self.xs_clean[i_gpu], n_hat)
		psnrs = 10*tf.log( 1.0 / tf.reduce_mean(tf.square(self.x_clean[i_gpu] - x_hat)) ) / tf.log(tf.constant(10.0))
		
		return loss, psnrs, x_hat


	def make_computational_graph(self):

		with tf.device('/cpu:0'):
			tower_grads = []	
			tower_psnrs = []
			global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
			#l_r = tf.train.exponential_decay(self.config.starter_learning_rate, global_step, self.config.decay_step, self.config.decay_rate, staircase=True)
			l_r = self.config.starter_learning_rate
			opt = tf.train.AdamOptimizer(l_r)
			for i in range(self.config.Ngpus):
				with tf.device('/gpu:' + str(i)):
					with tf.variable_scope('Tower_'+str(i)) as scope:
						#loss, psnr, x_hat = self.tower_loss(i)
						if i==0:
							loss, _, self.x0_hat = self.tower_loss(i)
						else:
							loss, _, self.x1_hat = self.tower_loss(i)
						tf.get_variable_scope().reuse_variables()	
						#grad_and_vars_final = opt.compute_gradients(loss, var_list=self.opt_vars)
						grad_and_vars_final = opt.compute_gradients(loss)
						grad_and_vars_final = [(g, v) for g, v in grad_and_vars_final if g is not None]
						tower_grads.append(grad_and_vars_final)
						#tower_psnrs.append(psnr)

			grads_avg = self.average_gradients(tower_grads)

			if self.config.grad_accum == 1:
				# no gradient aggregation
				self.apply_gradient_op = opt.apply_gradients(grads_avg, global_step=global_step)
			else:
				# gradient aggregation
				# create a copy of all trainable variables with `0` as initial values
				self.accum_vars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in self.dn_vars] 
				# create a op to initialize all accums vars
				self.zero_accum_vars = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
				# collect the batch gradient into accumulated vars	
				self.accum_op = [self.accum_vars[i].assign_add(batch_grad_var[0]) if batch_grad_var[0] is not None else self.accum_vars[i].assign_add(tf.zeros_like(self.accum_vars[i])) for i, batch_grad_var in enumerate(grads_avg)]
				# apply accums gradients 
				grad_and_vars_final_aggr = [(self.accum_vars[i], batch_grad_var[1]) if batch_grad_var[0] is not None else (None, batch_grad_var[1]) for i, batch_grad_var in enumerate(grads_avg)]
				self.apply_gradient_op = opt.apply_gradients(grad_and_vars_final_aggr, global_step=global_step)


			#self.psnr = tf.reduce_mean(tf.stack(tower_psnrs))


	def do_variables_init(self):
		
		with tf.device('/cpu:0'): 
			self.sess.run(tf.global_variables_initializer())


	def save_model(self, path):

		saver = tf.train.Saver()
		saver.save(self.sess, path)


	def restore_model_partial(self, path):

		vars_to_exclude=[]

		#vars_to_exclude=vars_to_exclude+[vv for vv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "Adam" in vv.name]

		vars_to_exclude=vars_to_exclude+self.accum_vars

		vars_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[varname.name for varname in vars_to_exclude])
		saver = tf.train.Saver(var_list=vars_to_restore)
		saver.restore(self.sess, path)


	def restore_model(self, path):

		saver = tf.train.Saver()
		saver.restore(self.sess, path)


	def fit(self, data_clean, data_noisy, iter_no, local_mask):

		feed_dict = {self.x_clean: data_clean, self.x_noisy: data_noisy, self.is_training: True, self.local_mask: local_mask}
		
		# if iter_no % self.config.summaries_every_iter == 0:
		# 	_ , summaries_train = self.sess.run((self.apply_gradient_op, self.summaries), feed_dict = feed_dict)
		# 	self.train_summaries_writer.add_summary(summaries_train, iter_no)
		# else:
		# 	self.sess.run(self.apply_gradient_op, feed_dict = feed_dict)

		self.sess.run(self.apply_gradient_op, feed_dict = feed_dict)


	def fit_gradaggr(self, data_clean, data_noisy, iter_no, local_mask):
			
		microbatch_size = data_clean.shape[0]/self.config.grad_accum
		for batch_iter in range(0, data_clean.shape[0], microbatch_size):

		 	data_clean_one = data_clean[batch_iter:(batch_iter+microbatch_size),:,:,:]
		 	data_noisy_one = data_noisy[batch_iter:(batch_iter+microbatch_size),:,:,:]

			feed_dict = {self.x_clean: data_clean_one, self.x_noisy: data_noisy_one, self.is_training: True, self.local_mask: local_mask}

			if batch_iter == 0:
				self.sess.run(self.zero_accum_vars, feed_dict = feed_dict)

			self.sess.run(self.accum_op, feed_dict = feed_dict)

		# if iter_no % self.config.summaries_every_iter == 0:
		# 	_ , summaries_train = self.sess.run((self.apply_accum, self.summaries), feed_dict = feed_dict)
		# 	self.train_summaries_writer.add_summary(summaries_train, iter_no)
		# else:
		# 	self.sess.run(self.apply_accum, feed_dict = feed_dict)

		self.sess.run(self.apply_gradient_op, feed_dict = feed_dict)
					

	def validate(self, data_clean, data_noisy, iter_no, local_mask):

		feed_dict = {self.x_clean: data_clean, self.x_noisy: data_noisy, self.is_training: False, self.local_mask: local_mask}

		summaries_val = self.sess.run(self.summaries, feed_dict = feed_dict)
		self.val_summaries_writer.add_summary(summaries_val, iter_no)


	def denoise(self, data_noisy, local_mask):

		feed_dict = {self.x_noisy: data_noisy, self.is_training: False, self.local_mask: local_mask}

		if self.config.Ngpus==1:
			x_hat_list = self.sess.run((self.x0_hat,), feed_dict = feed_dict)
		else:
			x_hat_list = self.sess.run((self.x0_hat, self.x1_hat), feed_dict = feed_dict)


		denoised_batch = np.concatenate(x_hat_list, axis=0)

		return denoised_batch