#!/home/valsesia/tensorflow-python2.7/bin/python
import os
import os.path as osp
import numpy as np
import shutil
import sys

from config import Config
from net import NET

import scipy.io as sio

import h5py

import argparse
from PIL import Image
import glob



def get_shifted_patches(noisy_image, shift_x, shift_y):

	img_Nrows, img_Ncols, img_Nbands = noisy_image.shape

	x_point = np.arange(0+shift_x, img_Ncols, config.patch_size[1]);
	y_point = np.arange(0+shift_y, img_Nrows, config.patch_size[0]);

	if x_point[-1]+config.patch_size[1] > img_Ncols:
		x_point = x_point[:-1]
	if y_point[-1]+config.patch_size[0] > img_Nrows:
		y_point = y_point[:-1]

	Npatches = len(x_point)*len(y_point);

	iter_no = 0;
	PallQ = np.zeros((Npatches, config.patch_size[1], config.patch_size[0], config.patch_size[2]));
	for xx in x_point:
		for yy in y_point:
			PallQ[iter_no,:,:,:] = noisy_image[yy:yy+config.patch_size[0], xx:xx+config.patch_size[1], :];
			iter_no = iter_no+1;	

	return PallQ	


def rebuild_image(patches, config, shift_x, shift_y, img_shape):
	
	img_Nrows = img_shape[0]
	img_Ncols = img_shape[1]
	img_Nbands = img_shape[2]

	x_point = np.arange(0+shift_x, img_Ncols, config.patch_size[1]);
	y_point = np.arange(0+shift_y, img_Nrows, config.patch_size[0]);

	if x_point[-1]+config.patch_size[1] > img_Ncols:
		x_point = x_point[:-1]
	if y_point[-1]+config.patch_size[0] > img_Nrows:
		y_point = y_point[:-1]

	x_hat = np.zeros(img_shape)
	iter_no=0;
	for xx in x_point:
		for yy in y_point:
			x_hat[yy:yy+config.patch_size[0], xx:xx+config.patch_size[1], :] = patches[iter_no,:,:,:];
			iter_no = iter_no+1;

	return x_hat



parser = argparse.ArgumentParser()
parser.add_argument('--start_iter', type=int, default=1, help='Start iteration (ex: 10001)')
parser.add_argument('--train_data_dir', default='', help='Training data directory')
parser.add_argument('--val_data_file', default='', help='Validation data file')
parser.add_argument('--test_data_file', default='', help='Testing data file')
parser.add_argument('--log_dir', default='', help='Tensorboard log directory')
parser.add_argument('--save_dir', default='', help='Trained model directory')
param = parser.parse_args()


# import config
config = Config()
config.log_dir = param.log_dir
config.save_dir = param.save_dir
config.train_data_dir = param.train_data_dir
config.val_data_file = param.val_data_file


# import train data
#with h5py.File(config.train_data_file) as dataset:
#	patches_clean = (dataset['Pall_clean'].value.astype(np.float).T)/255.0
#	patches_clean = patches_clean[:,:,:,np.newaxis]
#	#patches_noisy = (dataset['Pall_noisy'].value.astype(np.float).T)/255.0
#	#patches_noisy = patches_noisy[:,:,:,np.newaxis]
all_imgs=[]
all_imgs_noisy=[]
for train_image_file in glob.iglob(config.train_data_dir+'/*.png'):
	img = Image.open(train_image_file)
	img.load()
	cur_img = np.asarray(img, dtype="float32")
	if cur_img.shape[0] > cur_img.shape[1]:
		all_imgs = all_imgs + [cur_img]
	else:
		all_imgs = all_imgs + [np.rot90(cur_img)]

# import val data
#with h5py.File(config.val_data_file) as dataset_val:
#	patches_clean_val = (dataset_val['Pall_clean'].value.astype(np.float).T)/255.0
#	clean_val_batch = patches_clean_val[:,:,:,np.newaxis]
#	np.random.seed(47)
#	noisy_val_batch = clean_val_batch +  np.random.normal(0, config.sigma, clean_val_batch.shape)/255.0

# import test clean image, make it noisy
img = Image.open(param.test_data_file)
img.load()
clean_image = np.asarray( img, dtype="float32" )
np.random.seed(47)
noisy_image = clean_image + np.random.normal(0, config.sigma, clean_image.shape)
noisy_image = noisy_image/255.0
if len(clean_image.shape)==2:
	noisy_image = noisy_image[:,:,np.newaxis]

#for 48x48 patches
noisy_image=noisy_image[:420,:420]

# precompute mask for local neighborhood
local_mask = np.ones([config.N, config.N])
for ii in range(config.N):
	if ii==0:
		local_mask[ii, (ii+1, ii+config.patch_size[1], ii+config.patch_size[1]+1)] = 0 # top-left
	elif ii==config.N-1:
		local_mask[ii, (ii-1, ii-config.patch_size[1], ii-config.patch_size[1]-1)] = 0 # bottom-right
	elif ii==config.patch_size[0]-1:
		local_mask[ii, (ii-1, ii+config.patch_size[1], ii+config.patch_size[1]-1)] = 0 # top-right
	elif ii==config.N-config.patch_size[0]:
		local_mask[ii, (ii+1, ii-config.patch_size[1], ii-config.patch_size[1]+1)] = 0 # bottom-left
	elif ii<config.patch_size[0]-1 and ii>0:
		local_mask[ii, (ii+1, ii-1, ii+config.patch_size[1]-1, ii+config.patch_size[1], ii+config.patch_size[1]+1)] = 0 # first row
	elif ii<config.N-1 and ii>config.N-config.patch_size[0]:
		local_mask[ii, (ii+1, ii-1, ii-config.patch_size[1]-1, ii-config.patch_size[1], ii-config.patch_size[1]+1)] = 0 # last row
	elif ii%config.patch_size[1]==0:
		local_mask[ii, (ii+1, ii-config.patch_size[1], ii+config.patch_size[1], ii-config.patch_size[1]+1, ii+config.patch_size[1]+1)] = 0 # first col
	elif ii%config.patch_size[1]==config.patch_size[1]-1:
		local_mask[ii, (ii-1, ii-config.patch_size[1], ii+config.patch_size[1], ii-config.patch_size[1]-1, ii+config.patch_size[1]-1)] = 0 # last col
	else:
		local_mask[ii, (ii+1, ii-1, ii-config.patch_size[1], ii-config.patch_size[1]+1, ii-config.patch_size[1]-1, ii+config.patch_size[1], ii+config.patch_size[1]+1, ii+config.patch_size[1]-1)] = 0
local_mask = local_mask[np.newaxis,:,:]

# init model
model = NET(config)
model.do_variables_init()

if param.start_iter==1:
	start_iter = 0
	config.N_iter = config.N_iter+1
else:
	start_iter = param.start_iter
	model.restore_model(config.save_dir+'model.ckpt')
	print 'Resuming training from iter %d' % start_iter


np.random.seed(845608)

# training
Nimages = len(all_imgs)
for iter_no in range(start_iter, config.N_iter):

	# refresh training set every 200000 patches
	#if iter_no==start_iter or iter_no % (200000/config.batch_size) == 0:
	if iter_no==start_iter or iter_no % 10000 == 0:
		Npatches_per_image = 200
		patches_clean = np.zeros([Nimages*Npatches_per_image, config.patch_size[0], config.patch_size[1]])
		tot_p=0
		for img_i in range(Nimages):
			xsize = all_imgs[img_i].shape[1]
			ysize = all_imgs[img_i].shape[0]
			x_point = np.random.randint(xsize-config.patch_size[1], size=Npatches_per_image)
			y_point = np.random.randint(ysize-config.patch_size[0], size=Npatches_per_image)
			for p in range(Npatches_per_image):
				patches_clean[tot_p,:,:] = all_imgs[img_i][y_point[p]:(y_point[p]+config.patch_size[0]), x_point[p]:(x_point[p]+config.patch_size[1])]
				tot_p=tot_p+1
		patches_clean = patches_clean[:,:,:,np.newaxis]/255.0
			
	pos = np.random.choice(patches_clean.shape[0], size=config.batch_size)
	clean_batch = patches_clean[pos,:,:,:]
	noisy_batch = clean_batch + np.random.normal(0, config.sigma, clean_batch.shape)/255.0

	# train
	if config.grad_accum == 1:
		model.fit(clean_batch, noisy_batch, iter_no, local_mask)
	else:
		model.fit_gradaggr(clean_batch, noisy_batch, iter_no, local_mask)

	# validate
	#if iter_no % config.validate_every_iter == 0:
	#	model.validate(clean_val_batch, noisy_val_batch, iter_no, local_mask)

	# save model
	if iter_no % config.save_every_iter == 0:
		model.save_model(config.save_dir+'model.ckpt')
		with open(config.log_dir+'start_iter', "w") as text_file:
			text_file.write("%d" % iter_no)

	# backup model
	if iter_no % config.save_every_iter == 0:
		os.mkdir(config.save_dir+str(iter_no))
		model.save_model(config.save_dir+str(iter_no)+'/model.ckpt')

	# test
	if iter_no % config.test_every_iter == 0:
		
		shift_step_size = (42,42)
		cnt_img = np.zeros_like(noisy_image)
		x_hat = np.zeros_like(noisy_image)
						
		bad_data = get_shifted_patches(noisy_image, 0, 0)

		for b in range(0,bad_data.shape[0],config.batch_size):

			noisy_batch = bad_data[b:(b+config.batch_size),:,:,:]
			patches_denoised = model.denoise(noisy_batch, local_mask)
						
			if b==0:
				x_dn = patches_denoised+0.0
			else:	
				x_dn = np.concatenate( (x_dn, patches_denoised), axis=0 )

		cnt_img = cnt_img + rebuild_image(np.ones_like(x_dn), config, 0, 0, noisy_image.shape)
		x_hat = x_hat + rebuild_image(x_dn, config, 0, 0, noisy_image.shape)

		x_hat = (x_hat / cnt_img.astype(np.double))*255.0

		psnr = np.mean( 10*np.log10( 255.0*255.0 / np.mean(np.square(clean_image[:420,:420] - x_hat[:,:,0])) ) )
		print '[Iter %d] PSNR: %.2f' % (iter_no, psnr)
		with open(config.log_dir+'test_PSNR_newdataset.txt', "a") as psnr_file:
			 psnr_file.write('[Iter %d] PSNR: %.2f\n' % (iter_no, psnr))
		#img = Image.fromarray( np.asarray( np.clip(x_hat[:,:,0],0,255), dtype="uint8"), "L" )
		#img.save( config.save_dir+'/denoised.png' )
