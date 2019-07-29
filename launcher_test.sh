#!/bin/bash

models=("gcdn")
sigmas=("25")

this_folder=`pwd`

test_image_dir="$this_folder/Dataset/Gray/testset/Set12/"
for model in ${models[@]}; do
	for sigma in ${sigmas[@]}; do

		denoised_dir="$this_folder/Results/$model/$sigma/denoised_images/"
		model_dir="$this_folder/Results/$model/$sigma/saved_models/"

		CUDA_VISIBLE_DEVICES=0 python "Code/$model/test_conv_batched.py" --model_dir $model_dir --test_image_dir $test_image_dir --denoised_dir $denoised_dir --sigma $sigma --skip True
		
	done
done

