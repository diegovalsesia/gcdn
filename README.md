# Deep Graph-Convolutional Image Denoising

Code for paper "Deep Graph-Convolutional Image Denoising" by Diego Valsesia, Giulia Fracastoro, Enrico Magli.

BibTex references for journal and conference versions:

```
@article{valsesia2019deep,
  title={Deep Graph-Convolutional Image Denoising},
  author={Valsesia, Diego and Fracastoro, Giulia and Magli, Enrico},
  journal={arXiv preprint arXiv:1907.08448},
  year={2019}
}

@inproceedings{ValsesiaICIP19,
  title     = {Image Denoising with Graph-Convolutional Neural Networks},
  author    = {Valsesia, Diego and Fracastoro, Giulia and Magli, Enrico},
  year      = {2019},
  booktitle={2019 26th IEEE International Conference on Image Processing (ICIP)}
}

```

# Tested hardware configuration
```
CPU: AMD Ryzen 1 1700
RAM: 32 GB
GPUs: 2x Nvidia Quadro P6000 (24 GB)
```
The code requires a good amount of GPU memory. 2 GPUs with 24 GB are recommended for training. A single GPU is used for testing.


# Software requirements
- Python 2.7
- Tensorflow 1.12
- The packages listed in _requirements.txt_


# Code organization
Code is organized in the following directories
```
Code: one subfolder for each network model with all the source files;

Dataset: training ad testing images;

log_dir: logs for training process; if a start_iter file is found, training will resume from the iteration number written in the file after loading a model.

Results: saved_models directories contain checkpoints with trained models; denoised_images is the defualt path for output images.
```


# Parameters
File _config.py_ contains all the parameters used by the training and testing process.
```
patch_size : size of patches used in training
pre_n_layers : number of conv2d layers is the preprocessing block
pregconv_n_layers : number of gconv layers is the preprocessing block
hpf_n_layers : number of gconv layers in the HPF block
lpf_n_layers : number of gconv layers in each LPF block
prox_n_layers : number of LPF blocks
Nfeat : number of feature maps, must be a multiple of 3
rank_theta : maximum rank in low-rank approximation of ECC   
stride : Nfeat/num where num is how many circulant versions of a row are used in the circulant ECC aproximation
min_nn : total receptive field of gconv layer = number of non-local neighbors + 8 local neighbors
batch_size : total batch size (each GPU sees batch_size/Ngpus)
grad_accum : if >1 gradients from multiple minibatches are accumulated before being applied
N_iter : total number of training iterations
starter_learning_rate : learning rate (for fixed learning rate) or starting learning rate (for decay)
end_learning_rate : final learning rate (for decay)
Ngpus : number of GPUs used for training (only tested for 1 or 2)
minisize : maximum size of testing image portion to be considered at a time (to limit memory consumption), must be a multiple of search\_window
search\_window : size of window around each pixel where nearest neighbors are computed at testing time
sigma : standard deviation of noise. Remember to set it correctly when testing!
```


# Training
- Copy your training images (*.png) to the _Dataset/Gray/trainset_ directory
- Training can be run with the launcher bash script:
```
./launcher_train.sh
```

Training is parallelized over 2 GPUs. Notice that during training the gconv layer will compute all pairwise distances between the feature vectors of pixels within the patch. 
A small testing routine is included in the training code for debugging, but uses patchwise testing which creates border artifacts, contrary to the actual fully-convolutional testing code. Suggestion: improved convergence is achieved by pretraining a shorter model with only 2 LPF blocks (lpf_n_layers=2) and then increase its length to 4 LPF blocks.


# Testing
Pretrained models are available for noise standard deviations 15,25,50. Notice that the results can be slightly different from paper due to noise seed, initialization and finetuning of number of non-local neighbors. Pretrained models are finetuned on 16 non-local neighbors, changing that requires finetuning. Testing can be run with the launcher bash script:
```
./launcher_test.sh
```

Testing only works on a single GPU. Testing is fully-convolutional, so any input size is accepted. Contrary to training, only the distances from the pixel to be estimated to those in a search window around it are computed in the gconv layer. 

Testing parameters are optimized for a GPU with 24GB of memory. A few things can be changed to reduce (or increase) memory usage:

- Tweak the number of parallel threads in _net\_conv2.py_ (increases computation time):
```
Line 213: 
ret_list = tf.map_fn(lambda feat: self.gconv_conv_inner(feat[0], name, in_feat, out_feat, stride_th1, stride_th2, compute_graph, return_graph, feat[1]), [p, D], parallel_iterations=2, swap_memory=False) # needs 24GB
ret_list = tf.map_fn(lambda feat: self.gconv_conv_inner(feat[0], name, in_feat, out_feat, stride_th1, stride_th2, compute_graph, return_graph, feat[1]), [p, D], parallel_iterations=1, swap_memory=False) # lower memory usage
```
- Reduce the search window size (reduces output quality)
- Reduce the minisize value to let the network process a smaller crop of the image at a time (UNSTABLE, treat with care)