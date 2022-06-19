import json
import pandas as pd
import time
import datetime
import argparse

parser = argparse.ArgumentParser(description="experiment")
parser.add_argument(
    "--filename", type=str, help="json log file name")
args = parser.parse_args()

results = dict()
with open('/home/didwdidw/project/experiment_result/result_v3/simulation/resnet18-cifar10-records.json', 'r') as f:
    results = json.load(fp=f)

ground_truth_df = pd.DataFrame(list(results.items()), columns=['config', 'ground_truth_acc'])
print(ground_truth_df)

with open('/home/didwdidw/project/evaluations/results/extra/'+args.filename, 'r') as f:
    results = json.load(fp=f)

tune_log_df = pd.DataFrame(list(results.items()), columns=['config', 'iter'])
# print(tune_log_df)
# ground_truth_df["ground_truth_acc"] = ground_truth_df["ground_truth_acc"].apply(pd.to_numeric)
print(type(ground_truth_df["ground_truth_acc"][0]))


for index, row in tune_log_df.iterrows():
    row["config"] = row["config"] + "iter200"
tune_log_df["iter"] = pd.to_numeric(tune_log_df["iter"], errors='coerce')
print("tune log df\n", tune_log_df)
print("ground truth df\n", ground_truth_df)
evaluated_df = pd.merge(tune_log_df, ground_truth_df, on="config")
evaluated_df = evaluated_df.sort_values(by=['ground_truth_acc'], ascending=False)
print("evaluated df\n", evaluated_df)


ground_truth_df = ground_truth_df[ground_truth_df["config"].str.contains("iter200") == True]
ground_truth_df = ground_truth_df.sort_values(by=['ground_truth_acc'], ascending=False)
# search_space_df = ground_truth_df



weight_decay_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
momentum_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.93, 0.96, 0.99, 0.997]
search_space_df = pd.DataFrame(columns=['config', 'ground_truth_acc'])
for w in weight_decay_list:
    for lr in lr_list:
        for m in momentum_list:
            config_key="lr"+str(lr)+"momentum"+str(m)+"weight_decay"+str(w)+"iter200"
            search_space_df = search_space_df.append(ground_truth_df[ground_truth_df["config"] == config_key])
search_space_df = search_space_df.sort_values(by=['ground_truth_acc'], ascending=False)
# print(search_space_df)

# print(len(search_space_df[search_space_df["ground_truth_acc"]>0.95]))

experiment2_df = pd.merge(search_space_df, tune_log_df, on="config", how="left")
experiment2_df["iter"] = experiment2_df["iter"].fillna(0)
# print("experiment2 df\n", experiment2_df[:500].to_string())
total_iter = experiment2_df["iter"].sum()
cutoff = 0.05
print("total resource time", total_iter)
print("", len(experiment2_df[:int(len(search_space_df) * cutoff)][experiment2_df["iter"]==200]))
print("Fully trained the best trial in evaluated trials:", len(evaluated_df[:1][evaluated_df["iter"]==200]))
print("Fully trained trials the best trial in search space:", len(experiment2_df[:1][experiment2_df["iter"]==200]))
print("Fully trained trials in top "+str(0.01*100)+"% evaluated trials:", len(evaluated_df[:int(len(evaluated_df) * 0.01)][evaluated_df["iter"]==200]))
print("Fully trained trials in top "+str(0.01*100)+"% trials of search space:", len(experiment2_df[:int(len(search_space_df) * 0.01)][experiment2_df["iter"]==200]))
print("Fully trained trials in top "+str(cutoff*100)+"% evaluated trials:", len(evaluated_df[:int(len(evaluated_df) * cutoff)][evaluated_df["iter"]==200]))
print("Fully trained trials in top "+str(cutoff*100)+"% trials of search space:", len(experiment2_df[:int(len(search_space_df) * cutoff)][experiment2_df["iter"]==200]))
print("Fully trained trials in top "+str(2*cutoff*100)+"% evaluated trials:", len(evaluated_df[:int(len(evaluated_df) * 2*cutoff)][evaluated_df["iter"]==200]))
print("Fully trained trials in top "+str(2*cutoff*100)+"% trials of search space:", len(experiment2_df[:int(len(search_space_df) * 2*cutoff)][experiment2_df["iter"]==200]))
print("Fully trained trials in top "+str(4*cutoff*100)+"% evaluated trials:", len(evaluated_df[:int(len(evaluated_df) * 4*cutoff)][evaluated_df["iter"]==200]))
print("Fully trained trials in top "+str(4*cutoff*100)+"% trials of search space:", len(experiment2_df[:int(len(search_space_df) * 4*cutoff)][experiment2_df["iter"]==200]))
print("Fully trained trials in top "+str(6*cutoff*100)+"% evaluated trials:", len(evaluated_df[:int(len(evaluated_df) * 6*cutoff)][evaluated_df["iter"]==200]))
print("Fully trained trials in top "+str(6*cutoff*100)+"% trials of search space:", len(experiment2_df[:int(len(search_space_df) * 6*cutoff)][experiment2_df["iter"]==200]))