#!/bin/bash

# Change these variables 
#base_path=<path_to_experiment>
#exp_name="sigma0.1"
#dataset_name="TVSum"
#eval_method='avg'

# OR use arguments (example usage: bash evaluate_exp.sh <path_to_experiment> sigma0.1 TVSum avg)
base_path=$1
exp_name=$2
dataset_name=$3
eval_method=$4

exp_path="$base_path/$dataset_name/$exp_name"  #change this path if you use different structure for your directories inside the experiment

for i in {0..4}; do 
	path="$exp_path/logs/split$i"
	python exportTensorFlowLog.py $path $path
	results_path="$exp_path/results/split$i"
	python compute_fscores.py $results_path $dataset_name $eval_method
done
