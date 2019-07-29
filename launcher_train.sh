#!/bin/bash

models=("gcdn")
sigmas=("25")

this_folder=`pwd`

for model in ${models[@]}; do
	for sigma in ${sigmas[@]}; do
	
		# check if it is already trained or we want to retrain
		if [[ -f "$this_folder/log_dir/$model/$sigma/start_iter" ]]; then
			start_iter=`cat "$this_folder/log_dir/$model/$sigma/start_iter"`
		else
			start_iter=1
		fi
		
		train_data_dir="$this_folder/Dataset/Gray/trainset/"
		val_data_file="$this_folder/Dataset/Gray/validationset/val_patches48_sigma_25.mat"
		test_data_file="$this_folder/Dataset/Gray/testset/Set12/08.png"
		log_dir="$this_folder/log_dir/$model/$sigma/"
		save_dir="$this_folder/Results/$model/$sigma/saved_models/"
		CUDA_VISIBLE_DEVICES=0,1 python "Code/$model/main.py" --start_iter $start_iter --train_data_dir $train_data_dir --val_data_file $val_data_file --test_data_file $test_data_file --log_dir $log_dir --save_dir $save_dir


	done
done

