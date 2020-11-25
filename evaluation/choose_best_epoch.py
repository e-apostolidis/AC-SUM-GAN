import csv
import json
import sys
import torch
import numpy as np

''' Chooses the best F-score (among 100 epochs) based on a criterion (Reward & Actor_loss).
    Takes as input the path to .csv file with all the loss functions and a .txt file with the F-Scores (for each split).
    Prints a scalar that represents the average best F-score value.'''


def use_logs(logs_file, f_scores):
	losses = {}
	losses_names = []

	with open(logs_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for (i, row) in enumerate(csv_reader):
			if i == 0:
				for col in range(len(row)):
					losses[row[col]] = []
					losses_names.append(row[col])
			else:
				for col in range(len(row)):
					losses[losses_names[col]].append(float(row[col]))

	# criterion: Reward & Actor_loss
	actor = losses['actor_loss_epoch']
	reward = losses['reward_epoch']

	actor_t = torch.tensor(actor)
	reward_t = torch.tensor(reward)

	# Normalize values
	actor_t = abs(actor_t)
	actor_t = actor_t/max(actor_t)
	reward_t = reward_t/max(reward_t)

	product = (1-actor_t)*reward_t

	epoch = torch.argmax(product) + 1

	return np.round(f_scores[epoch], 2)

# with args (example usage: python choose_best_epoch.py <path_to_experiment> TVSum)
exp_path = sys.argv[1]
dataset = sys.argv[2]

# without args
'''exp_path = <path_to_experiment>
dataset = 'TVSum' '''

# For each 'sigma' value, compute the best F-Score of each split based on the criterion
all_fscores = np.zeros((5, 10), dtype=float)
for i, sigma in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
	path = exp_path+'/dataset/sigma'+str(sigma)  #change this path if you use different structure for your directories inside the experiment
	for split in range(0,5):
		results_file = path+'/results/split'+str(split)+'/f_scores.txt'
		logs_file = path+'/logs/split'+str(split)+'/scalars.csv'

		# read F-Scores
		with open(results_file) as f:
			f_scores = json.loads(f.read())	# list of F-Scores

		# best F-Score based on train logs
		all_fscores[split,i] = use_logs(logs_file, f_scores)

best_per_split = np.max(all_fscores, axis=1)
best_fscore = np.mean(best_per_split)
print(best_fscore)
