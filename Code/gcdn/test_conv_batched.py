#!/home/valsesia/tensorflow-python2.7/bin/python
import os
import os.path as osp
import numpy as np
import shutil
import sys

from config import Config
from net_conv2 import NET

import scipy.io as sio
from PIL import Image
import h5py

import argparse
import glob

from hashlib import sha256


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', help='Saved model directory')
parser.add_argument('--test_image_dir', default='', help='Directory with test images')
parser.add_argument('--denoised_dir', default='', help='Denoised image directory')
parser.add_argument('--sigma', type=int, default=25, help='Noise standard deviation')
parser.add_argument('--skip', type=bool, default=True, help='Skip images already in results directory')
param = parser.parse_args()


# import config
config = Config()
minisize = config.minisize
mini_image_shape = [minisize+config.search_window[0],minisize+config.search_window[0],1]
config.patch_size = mini_image_shape
config.N = config.patch_size[0]*config.patch_size[1]

local_mask = np.ones([config.searchN,])
ii = config.searchN/2
local_mask[ii+1] = 0
local_mask[ii-1] = 0
local_mask[ii-config.search_window[1]] = 0
local_mask[ii-config.search_window[1]+1] = 0
local_mask[ii-config.search_window[1]-1] = 0
local_mask[ii+config.search_window[1]] = 0
local_mask[ii+config.search_window[1]+1] = 0
local_mask[ii+config.search_window[1]-1] = 0

# init and restore model
# trick to realign seed for S matrix in net
np.random.seed(47)
tmp = np.zeros([512,512]) + np.random.normal(0, 25, [512,512])
model = NET(config)
model.do_variables_init()
model.restore_model(param.model_dir+'model.ckpt')


# loop over all images
for test_image_file in glob.iglob(param.test_image_dir+'/*.png'):

	output_name = test_image_file.split('/')[-1].split('.')[0]
	
	if osp.isfile(param.denoised_dir+'/'+output_name+'_denoised.mat'):
		print "Skipping " + output_name
		continue
	else:
		print "Processing " + output_name

		# import clean image, make it noisy
		img = Image.open(test_image_file)
		img.load()
		clean_image = np.asarray( img, dtype="float32" )
		np.random.seed(47)
		noisy_image = clean_image + np.random.normal(0, param.sigma, clean_image.shape)
		noisy_image = noisy_image/255.0
		if len(clean_image.shape)==2:
			noisy_image = noisy_image[:,:,np.newaxis]

		# test
		noisy_image = noisy_image[np.newaxis,:,:,:]
		x_hat = np.zeros_like(noisy_image)
		for sub_y in range(0,noisy_image.shape[1],minisize):
			for sub_x in range(0,noisy_image.shape[2],minisize):
				#print sub_x, sub_y
				if sub_y == 0:
					if sub_x == 0:
						#0,0
		 				tmp_dn = model.denoise( noisy_image[:,sub_y:(sub_y+minisize+config.search_window[0]), sub_x:(sub_x+minisize+config.search_window[1]),:], local_mask )
		 				x_hat[:,sub_y:(sub_y+minisize), sub_x:(sub_x+minisize),:] = tmp_dn[:,0:(minisize), 0:(minisize),:]
					else:
						#0,x
						if sub_x + minisize + config.search_window[1]/2 >= noisy_image.shape[2]:
							diff = sub_x + minisize - noisy_image.shape[2]
							tmp_dn = model.denoise( noisy_image[:,sub_y:(sub_y+minisize+config.search_window[0]), (sub_x-diff-config.search_window[1]):,:], local_mask )
							x_hat[:,sub_y:(sub_y+minisize), sub_x:,:] = tmp_dn[:,0:(minisize), (diff+config.search_window[1]):,:]
						else:
		 					tmp_dn = model.denoise( noisy_image[:,sub_y:(sub_y+minisize+config.search_window[0]), (sub_x-config.search_window[1]/2-1):(sub_x+minisize+config.search_window[1]/2),:], local_mask )
		 					x_hat[:,sub_y:(sub_y+minisize), sub_x:(sub_x+minisize),:] = tmp_dn[:,0:(minisize), (config.search_window[1]/2+1):(minisize+config.search_window[1]/2+1),:]
				else:
					if sub_y + minisize + config.search_window[0]/2 >= noisy_image.shape[1]:
						diff = sub_y + minisize - noisy_image.shape[1]
						if sub_x == 0:
							tmp_dn = model.denoise( noisy_image[:,(sub_y-diff-config.search_window[0]):, sub_x:(minisize+config.search_window[1]),:], local_mask )
							x_hat[:,sub_y:, sub_x:(sub_x+minisize),:] = tmp_dn[:,(diff+config.search_window[0]):, 0:(minisize),:]	
						else:
							if sub_x + minisize + config.search_window[1]/2 >= noisy_image.shape[2]:
								diffx = sub_x + minisize - noisy_image.shape[2]
								tmp_dn = model.denoise( noisy_image[:,(sub_y-diff-config.search_window[0]):, (sub_x-diffx-config.search_window[1]):,:], local_mask )
								x_hat[:,sub_y:, sub_x:,:] = tmp_dn[:,(diff+config.search_window[0]):, (diffx+config.search_window[1]):,:]
							else:
								tmp_dn = model.denoise( noisy_image[:,(sub_y-diff-config.search_window[0]):, (sub_x-config.search_window[1]/2-1):(sub_x+minisize+config.search_window[1]/2),:], local_mask )
								x_hat[:,sub_y:, sub_x:(sub_x+minisize),:] = tmp_dn[:,(diff+config.search_window[0]):, (config.search_window[1]/2+1):(minisize+config.search_window[1]/2+1),:]	
					else:
						if sub_x == 0:
							#y,0
		 					tmp_dn = model.denoise( noisy_image[:,(sub_y-config.search_window[0]/2-1):(sub_y+minisize+config.search_window[0]/2), sub_x:(minisize+config.search_window[1]),:], local_mask )
		 					x_hat[:,sub_y:(sub_y+minisize), sub_x:(sub_x+minisize),:] = tmp_dn[:,(config.search_window[0]/2+1):(config.search_window[0]/2+minisize+1), 0:(minisize),:]	
						else:
							if sub_x + minisize + config.search_window[1]/2 >= noisy_image.shape[2]:		
								diff = sub_x + minisize - noisy_image.shape[2]
								tmp_dn = model.denoise( noisy_image[:,(sub_y-config.search_window[0]/2-1):(sub_y+minisize+config.search_window[0]/2), (sub_x-diff-config.search_window[1]):,:], local_mask )
								x_hat[:,sub_y:(sub_y+minisize), sub_x:,:] = tmp_dn[:,(config.search_window[0]/2+1):(config.search_window[0]/2+minisize+1), (diff+config.search_window[1]):,:]
							else:
								#x,y
		 						tmp_dn = model.denoise( noisy_image[:,(sub_y-config.search_window[0]/2-1):(sub_y+minisize+config.search_window[0]/2), (sub_x-config.search_window[1]/2-1):(sub_x+minisize+config.search_window[1]/2),:], local_mask )
		 						x_hat[:,sub_y:(sub_y+minisize), sub_x:(sub_x+minisize),:] = tmp_dn[:,(config.search_window[0]/2+1):(config.search_window[0]/2+minisize+1), (config.search_window[1]/2+1):(minisize+config.search_window[1]/2+1),:]

					
		x_hat = x_hat[0,:,:,:]*255.0

		img = Image.fromarray( np.asarray( np.clip(x_hat[:,:,0],0,255), dtype="uint8"), "L" )
		img.save( param.denoised_dir+'/'+output_name+'_denoised.png' )
		sio.savemat(param.denoised_dir+'/'+output_name+'_denoised.mat',{'x_hat': x_hat})
