import numpy as np
import tensorflow as tf
import time
import random

from tensorflow.python.client import timeline

import scipy.io as sio


class NET:


	def __init__(self, config):

		self.config = config 
		self.N = config.N

		######### not running out gpu sources ##########
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = tf_config)
		
		######### profiling #############################
		#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#self.run_metadata = tf.RunMetadata()

		############ define variables ##################
		self.W = {}
		self.b = {}
		self.scale={}
		self.beta={}
		self.pop_mean={}
		self.pop_var={}
		self.alpha={}

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

			
		############ define placeholders ##############
		self.x_clean = tf.placeholder("float", [None, config.patch_size[0], config.patch_size[1], config.patch_size[2]], name="clean_image")
		self.x_noisy = tf.placeholder("float", [None, config.patch_size[0], config.patch_size[1], config.patch_size[2]], name="noisy_image")
		self.is_training = tf.placeholder(tf.bool, (), name="is_training")
		self.local_mask = tf.placeholder("float", [config.searchN,], name="local_mask")

		self.id_mat = 2*tf.eye(config.searchN)

		########### computational graph ###############
		self.__make_compute_graph()


		################## losses #####################
		self.__make_loss()

		################ optimizer ops ################
		#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		#with tf.control_dependencies(update_ops):
		
		#global_step = tf.Variable(0, trainable=False)
		#l_r = tf.train.exponential_decay(config.starter_learning_rate, global_step, config.decay_step, config.decay_rate, staircase=True)

		#self.opt = tf.train.AdamOptimizer(l_r)
		# create a copy of all trainable variables with `0` as initial values
		#self.accum_vars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in dn_vars] 
		# create a op to initialize all accums vars
		#self.zero_accum_vars = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
		# compute gradients for a batch
		#batch_grads_vars = self.opt.compute_gradients(self.loss, dn_vars)
		# collect the batch gradient into accumulated vars
		#self.accum_op = self.my_accum_op(batch_grads_vars)		
		#self.accum_op = [self.accum_vars[i].assign_add(batch_grad_var[0]) if batch_grad_var[0] is not None else self.accum_vars[i].assign_add(tf.zeros_like(self.accum_vars[i])) for i, batch_grad_var in enumerate(batch_grads_vars)]
		# apply accums gradients 
		#print [(self.accum_vars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)]
		#print batch_grads_vars
		#grad_and_vars_final = [(self.accum_vars[i], batch_grad_var[1]) if batch_grad_var[0] is not None else (None, batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)]
		#self.apply_accum = self.opt.apply_gradients(grad_and_vars_final)
		#self.apply_accum = self.opt.apply_gradients(batch_grads_vars)
		self.opt = tf.train.AdamOptimizer(config.end_learning_rate).minimize(self.loss, var_list=self.dn_vars, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_TREE)		

		################# summaries ###################
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('PSNR', self.psnr)
		tf.summary.image('denoised_image',tf.expand_dims(self.x_hat[0,:,:,:],0))
		tf.summary.image('noisy_image',tf.expand_dims(self.x_noisy[0,:,:,:],0))
		tf.summary.image('clean_image',tf.expand_dims(self.x_clean[0,:,:,:],0))
		self.summaries = tf.summary.merge_all()
		# Check if log_dir exists, if so delete contents
		#if tf.gfile.Exists(self.config.log_dir):
		#	tf.gfile.DeleteRecursively(self.config.log_dir)
		#	tf.gfile.MkDir(self.config.log_dir+'train/')
		#	tf.gfile.MkDir(self.config.log_dir+'val/')
		self.train_summaries_writer = tf.summary.FileWriter(self.config.log_dir+'train/', self.sess.graph)
		self.val_summaries_writer = tf.summary.FileWriter(self.config.log_dir+'val/', self.sess.graph)


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


	# same as new tf.roll but only for 3D input and axis=2
	def myroll(self, h, shift=0, axis=2):
			
		h_len = h.get_shape()[2]
		return tf.concat([h[:,:,h_len-shift:], h[:,:,:h_len-shift]], axis=2)


	def gconv_conv(self, h, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False, D=[]):

		M = 2*self.config.search_window[0]-1
		#M = self.config.patch_size[0]-1

		if compute_graph:
			D = tf.zeros([(self.config.minisize/self.config.search_window[0]+1)**2,M,M])
			#D = tf.zeros([4,M,M])

		padborder1 = tf.constant([[0,0],[1, 1],[1, 1],[0, 0]])
		padborder = tf.constant([[0,0],[self.config.search_window[0]/2-1, self.config.search_window[0]/2-1],[self.config.search_window[1]/2-1, self.config.search_window[1]/2-1],[0, 0]])

		h = tf.reshape(h, [1, self.config.patch_size[0], self.config.patch_size[1], in_feat]) # (1,N,dlm1) --> (1,X,Y,dlm1)	
		h = tf.pad(h, padborder1, "SYMMETRIC")
		h = tf.pad(h, padborder, "CONSTANT", constant_values=47000)

		p = tf.image.extract_image_patches(h, ksizes=[1, M, M, 1], strides=[1,M-self.config.search_window[0]+1,M-self.config.search_window[1]+1,1], rates=[1,1,1,1], padding="VALID") # (1,M,M,dlm1*4)
		p = tf.reshape(p,[-1, M, M, in_feat]) 

		ret_list = tf.map_fn(lambda feat: self.gconv_conv_inner(feat[0], name, in_feat, out_feat, stride_th1, stride_th2, compute_graph, return_graph, feat[1]), [p, D], parallel_iterations=2, swap_memory=False) # (4, M, dl)

		xs = tf.reshape(ret_list[0], [-1, M-self.config.search_window[0]+1, M-self.config.search_window[1]+1, out_feat])
		xsl = []
		for rr in range(0, (self.config.minisize/self.config.search_window[0]+1)**2, (self.config.minisize/self.config.search_window[0]+1)):
			xh = tf.unstack(xs[rr:(rr+self.config.minisize/self.config.search_window[0]+1),:,:,:])
			xsl = xsl+[tf.concat(xh, axis=1)]
		xs = tf.concat(xsl, axis=0)
		#xh1 = tf.concat([xs[0,:,:,:], xs[1,:,:,:]], axis=1)
		#xh2 = tf.concat([xs[2,:,:,:], xs[3,:,:,:]], axis=1)
		#xs = tf.concat([xh1, xh2], axis=0)
		##xs = tf.reshape(xs, [(self.config.minisize/self.config.search_window[0]+1)*(M-self.config.search_window[0]+1),(self.config.minisize/self.config.search_window[0]+1)*(M-self.config.search_window[0]+1), out_feat])
		xs = tf.reshape(xs, [1, -1, out_feat])

		if return_graph:
			return xs, ret_list[1]
		else:
			return xs


	def gconv_conv_inner(self, h, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False, D=[]):

		h = tf.expand_dims(h, 0) # (1,M,dl)
		p = tf.image.extract_image_patches(h, ksizes=[1, self.config.search_window[0], self.config.search_window[1], 1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID") # (1,X,Y,dlm1*W)
		p = tf.reshape(p,[-1, self.config.search_window[0], self.config.search_window[1], in_feat]) 
		p = tf.reshape(p,[-1, self.config.searchN, in_feat]) # (N,W,dlm1)

		if compute_graph:
			D = tf.map_fn(lambda feat: self.gconv_conv_inner2(feat), tf.reshape(p,[self.config.search_window[0],self.config.search_window[1],self.config.searchN, in_feat]), parallel_iterations=16, swap_memory=False) # (B,N/B,W)
			D = tf.reshape(D,[-1, self.config.searchN]) # (N,W)

		_, top_idx = tf.nn.top_k(-D, self.config.min_nn+1) # (N, d+1)
		#top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:,0],1), [1, self.config.min_nn[i]]), [-1])
		top_idx2 = tf.tile(tf.expand_dims(top_idx[:,0],1), [1, self.config.min_nn-8]) # (N, d)
		#top_idx = tf.reshape(top_idx[:,1:],[-1]) # (N*d,)
		top_idx = top_idx[:,9:] # (N, d)

		x_tilde1 = tf.batch_gather(p, top_idx) # (N, d, dlm1)	
		x_tilde1 = tf.reshape(x_tilde1, [-1, in_feat]) # (K, dlm1)
		x_tilde2 = tf.batch_gather(p, top_idx2) # (N, d, dlm1)
		x_tilde2 = tf.reshape(x_tilde2, [-1, in_feat]) # (K, dlm1)

		labels = x_tilde1 - x_tilde2 # (K, dlm1)
		d_labels = tf.reshape( tf.reduce_sum(labels*labels, 1), [-1, self.config.min_nn-8]) # (N, d)

		name_flayer = name + "_flayer0"
		labels = tf.nn.leaky_relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])
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

		x = tf.matmul(theta1, tf.expand_dims(x_tilde1,2)) # (K, R, 1)
		x = tf.multiply(x, thetal) # (K, R, 1)
		x = tf.matmul(theta2, x, transpose_a=True)[:,:,0] # (K, dl)

		x = tf.reshape(x, [-1, self.config.min_nn-8, out_feat]) # (N, d, dl)
		x = tf.multiply(x, tf.expand_dims(tf.exp(-tf.div(d_labels,10)),2)) # (N, d, dl)
		x = tf.reduce_mean(x, 1) # (N, dl)

		x = tf.expand_dims(x,0) # (1, N, dl)

		return [x, D]


	def gconv_conv_inner2(self, p):

		p = tf.cast(p, tf.float64)
		# find central pixel
		p_central = p[:,self.config.searchN/2,:] # (N,F)
		# distances between central pixels and all other pixels
		central_norm = tf.reduce_sum(p_central*p_central, 1) # (N,)
		all_norms = tf.reduce_sum(p*p, 2) # (N,W)
		D = tf.abs( tf.expand_dims(central_norm,1) + all_norms - 2*tf.matmul(p, tf.expand_dims(p_central,2))[:,:,0] ) # (N,W)
		p = tf.cast(p, tf.float32) 
		D = tf.cast(D, tf.float32)	
		D = tf.multiply(D, self.local_mask)
		D = D - tf.expand_dims(self.id_mat[:,self.config.searchN/2], 0)

		return D


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


	def lnl_aggregation(self, h_l, h_nl, b):
		
		return tf.div(h_l + h_nl, 2) + b
		#return h_l + b


	def __make_compute_graph(self):

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
				h3_nl = self.gconv_conv(h3, name_block+"3_nl_"+str(i), self.config.pre_Nfeat, self.config.pre_Nfeat, self.config.stride_pregconv, self.config.stride_pregconv, compute_graph=True, return_graph=False)
				h3_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h3,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.pre_Nfeat]), paddings3, "REFLECT"), self.W[name_block+"3_l_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.pre_Nfeat])
				h3 = self.lnl_aggregation(h3_l, h3_nl, self.b[name_block+"3_"+str(i)])
				h5_nl = self.gconv_conv(h5, name_block+"5_nl_"+str(i), self.config.pre_Nfeat, self.config.pre_Nfeat, self.config.stride_pregconv, self.config.stride_pregconv, compute_graph=True, return_graph=False)
				h5_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h3,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.pre_Nfeat]), paddings3, "REFLECT"), self.W[name_block+"5_l_"+str(i)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.pre_Nfeat])
				h5 =  self.lnl_aggregation(h5_l, h5_nl, self.b[name_block+"5_"+str(i)])
				h7_nl = self.gconv_conv(h7, name_block+"7_nl_"+str(i), self.config.pre_Nfeat, self.config.pre_Nfeat, self.config.stride_pregconv, self.config.stride_pregconv, compute_graph=True, return_graph=False)
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
					h_hpf_nl, D = self.gconv_conv(h_hpf, name_block+"_nl_"+str(i), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=True, return_graph=True)
				else:		
					h_hpf_nl = self.gconv_conv(h_hpf, name_block+"_nl_"+str(i), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=False, return_graph=False, D=D)
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
						h_nl, D = self.gconv_conv(h, name_block+str(i)+"_nl_"+str(j), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=True, return_graph=True)
					else:
						h_nl = self.gconv_conv(h, name_block+str(i)+"_nl_"+str(j), self.config.Nfeat, self.config.Nfeat, self.config.stride, self.config.stride, compute_graph=False, return_graph=False, D=D)		
					h_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+str(i)+"_l_"+str(j)], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.Nfeat])	
					h = self.lnl_aggregation(h_l, h_nl, self.b[name_block+str(i)+"_"+str(j)])
					h = self.batch_norm_wrapper(h, name_block+str(i)+"_"+str(j))
					h = tf.nn.leaky_relu(h)
				h = h + h_old

			# last
			name_block = "last"
			h_nl = self.gconv_conv(h, name_block+"_nl_0", self.config.Nfeat, self.config.patch_size[2], self.config.stride, self.config.stride, compute_graph=True, return_graph=False)
			h_l = tf.reshape(tf.nn.conv2d(tf.pad(tf.reshape(h,[-1, self.config.patch_size[0], self.config.patch_size[1], self.config.Nfeat]), paddings3, "REFLECT"), self.W[name_block+"_l_0"], strides=[1,1,1,1], padding="VALID"), [-1, self.N, self.config.patch_size[2]])	
			h = self.lnl_aggregation(h_l, h_nl, self.b[name_block+"_0"])

			h = tf.reshape(h, [-1, self.config.patch_size[0], self.config.patch_size[1], self.config.patch_size[2]])

			return h

		self.n_hat = noise_extract( self.x_noisy ) 
		self.x_hat = self.x_noisy - self.n_hat


	def __make_loss(self):
		
		# mse
		self.loss = tf.losses.mean_squared_error(self.x_noisy[self.config.search_window[0]/2:(self.config.patch_size[0]-self.config.search_window[0]/2), self.config.search_window[1]/2:(self.config.patch_size[1]-self.config.search_window[1]/2)]-self.x_clean[self.config.search_window[0]/2:(self.config.patch_size[0]-self.config.search_window[0]/2), self.config.search_window[1]/2:(self.config.patch_size[1]-self.config.search_window[1]/2)], self.n_hat[self.config.search_window[0]/2:(self.config.patch_size[0]-self.config.search_window[0]/2), self.config.search_window[1]/2:(self.config.patch_size[1]-self.config.search_window[1]/2)]) # discard border
		#self.snr = tf.reduce_mean( 10*tf.log( tf.reduce_sum(tf.square(self.x_clean), axis=[1,2,3]) / tf.reduce_sum(tf.square(self.x_clean - self.x_noisy), axis=[1,2,3]) ) ) / tf.log(tf.constant(10.0))
		self.psnr = tf.reduce_mean( 10*tf.log( 1.0 / tf.reduce_mean(tf.square(self.x_clean[self.config.search_window[0]/2:(self.config.patch_size[0]-self.config.search_window[0]/2), self.config.search_window[1]/2:(self.config.patch_size[1]-self.config.search_window[1]/2)] - self.x_hat[self.config.search_window[0]/2:(self.config.patch_size[0]-self.config.search_window[0]/2), self.config.search_window[1]/2:(self.config.patch_size[1]-self.config.search_window[1]/2)])) ) ) / tf.log(tf.constant(10.0))


	def do_variables_init(self):

		init = tf.global_variables_initializer()       
		self.sess.run(init)


	def save_model(self, path):

		saver = tf.train.Saver()
		saver.save(self.sess, path)


	def restore_model(self, path):

		saver = tf.train.Saver()
		saver.restore(self.sess, path)
		self.is_Init = True


	def fit(self, data_clean, data_noisy, iter_no, local_mask):

		feed_dict = {self.x_clean: data_clean, self.x_noisy: data_noisy, self.is_training: True, self.local_mask: local_mask}
		
		# self.sess.run(self.zero_accum_vars, feed_dict = feed_dict)
		# for batch_iter in range(self.config.grad_accum):
		# 	self.sess.run(self.accum_op, feed_dict = feed_dict)

		# if iter_no % self.config.summaries_every_iter == 0:
		# 	_ , summaries_train = self.sess.run((self.apply_accum, self.summaries), feed_dict = feed_dict)
		# 	self.train_summaries_writer.add_summary(summaries_train, iter_no)
		# else:
		# 	self.sess.run(self.apply_accum, feed_dict = feed_dict)

		if iter_no % self.config.summaries_every_iter == 0:
			_ , summaries_train = self.sess.run((self.opt, self.summaries), feed_dict = feed_dict)
			self.train_summaries_writer.add_summary(summaries_train, iter_no)
		else:
			self.sess.run(self.opt, feed_dict = feed_dict)
					

	def validate(self, data_clean, data_noisy, iter_no, local_mask):

		feed_dict = {self.x_clean: data_clean, self.x_noisy: data_noisy, self.is_training: False, self.local_mask: local_mask}

		summaries_val = self.sess.run(self.summaries, feed_dict = feed_dict)
		self.val_summaries_writer.add_summary(summaries_val, iter_no)


	def denoise(self, data_noisy, local_mask):

		feed_dict = {self.x_noisy: data_noisy, self.is_training: False, self.local_mask: local_mask}

		denoised_batch = self.sess.run(self.x_hat, feed_dict = feed_dict)

		return denoised_batch
