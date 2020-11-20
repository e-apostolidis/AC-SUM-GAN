# Run AC-SUM-GAN for 10 values of the regularization factor
bash run_splits.sh 0.1
bash run_splits.sh 0.2
bash run_splits.sh 0.3
bash run_splits.sh 0.4
bash run_splits.sh 0.5
bash run_splits.sh 0.6
bash run_splits.sh 0.7
bash run_splits.sh 0.8
bash run_splits.sh 0.9
bash run_splits.sh 1.0

# Run the evaluation script with the right arguments to compute the F-Scores and extract the loss values for each value of the regularization factor (sigma)
bash evaluate_exp.sh <path_to_experiment> sigma0.1 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.2 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.3 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.4 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.5 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.6 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.7 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.8 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma0.9 TVSum avg
bash evaluate_exp.sh <path_to_experiment> sigma1.0 TVSum avg

# Run the script that chooses the best epoch and therefore the final F-Score value
python choose_best_epoch.py <path_to_experiment> TVSum