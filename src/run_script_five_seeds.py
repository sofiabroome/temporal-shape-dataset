import csv
import argparse

import main


h_values = [2, 4]
# h_values = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48]

seeds = [42, 61]
# seeds = [42, 61, 47, 53, 57]

parser = argparse.ArgumentParser()

# load configurations
parser.add_argument('--model', '-m', help='convlstm | 3dconv | timesf')
parser.add_argument('--train_dataset', '-td', help='2dot | mnistbg')
parser.add_argument('--test_run', action='store_true',
                    help="quick test run")
parser.add_argument('--job_identifier', '-j', help='Unique identifier for run,'
                                                   'avoids overwriting model.')
parser.add_argument('--results_persist', action='store_true',
                    help="whether to save accuracies across repeated runs in a .csv or not")

args = parser.parse_args()

model = args.model
train_dataset = args.train_dataset

first_experiment = True

config_name = model + '.json'

if train_dataset == '2dot':
    config_folder = 'configs/train2dot/'
if train_dataset == 'mnistbg':
    config_folder = 'configs/trainmnistbg/'

config_path = config_folder + config_name

save_results_file = 'results/temporalshape_5seeds/' + model + '_train' + train_dataset + '.csv'

file = open(save_results_file, 'w')
writer = csv.writer(file)
title_data = [model, train_dataset]
writer.writerow(title_data)
file.close()

for h in h_values:
    file = open(save_results_file, 'a')
    writer = csv.writer(file)
    title_data = ["h=" + str(h)]
    writer.writerow(title_data)
    file.close()

    test_accuracies = [['best_val', 'test_1', 'test_2', 'test_3']]
    for seed in seeds:
        # test_accuracies_one_seed = [0.5,0.4,0.3,0.2]
        test_accuracies_one_seed = main.main(parser,
                                             hidden_units=h,
                                             config_path=config_path,
                                             seed=seed,
                                             first_experiment=first_experiment)
        first_experiment = False
        test_accuracies_one_seed = [round(float(elt), 4) for elt in test_accuracies_one_seed]
        test_accuracies.append(test_accuracies_one_seed)

    # Write the 4x5 table to the .csv for this specific seed and #hidden units
    to_csv = zip(*test_accuracies)
    file = open(save_results_file, 'a')
    writer = csv.writer(file)
    writer.writerows(to_csv)
    file.close()


