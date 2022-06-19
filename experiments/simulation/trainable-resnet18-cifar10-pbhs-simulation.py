import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
# from dataset import Cifar10Dataset
from tensorboardX import SummaryWriter
import torchvision
import os
import ray
from ray import tune
from ray.tune.schedulers.pbhs import PBHS
from ray.tune.schedulers import HyperBandScheduler
import random
import argparse
from models import *
import time
import json
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Cifar ResNet Example")
parser.add_argument(
    "--use-gpu",
    action="store_true",
    default=True,
    help="enables CUDA training")
parser.add_argument(
    "--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
parser.add_argument(
    "--gpus", type=int, help="the number of GPUs")
parser.add_argument(
    "--deadline", type=int, help="experiment time limit")

log_file_dir = '/home/didwdidw/project/evaluations/results/extra/'
log_file_name = "tune-log-pbhs-" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".json"
log_file_path = log_file_dir+log_file_name

train_info = dict()
starter = {"head": 0}
with open(log_file_path, 'w', encoding='utf-8') as record_file:
    json.dump(starter, fp=record_file)
    print("create json successfully")

class TrainCifarResNet(tune.Trainable):

    def setup(self, config):
        self.epoches = 0
        self.results = dict()

        with open('/home/didwdidw/project/experiment_result/result_v3/simulation/resnet18-cifar10-records.json', 'r') as f:
            self.results = json.load(fp=f)

        
        # self.config_lr=str(config.get("lr", 0.1))
        # self.config_momentum=str(config.get("momentum", 0.9))
        # self.config_weight_decay=str(config.get("weight_decay", 5e-4))
        self.config_lr=str(config.get("lr", 0.1)) if str(config.get("lr", 0.1)) != "1.0" else "1"
        self.config_momentum=str(config.get("momentum", 0.9))
        self.config_weight_decay=str(config.get("weight_decay", 5e-4)) if str(config.get("weight_decay", 5e-4)) != "1.0" else "1"
        
    def step(self):
        iter = str(self.epoches+1)
        hyperparams = "lr"+self.config_lr+"momentum"+self.config_momentum+"weight_decay"+self.config_weight_decay
        final_acc_key = hyperparams+"iter"+"200"
        
        config_key = hyperparams+"iter"+iter
        acc = self.results[config_key]

        # train_info = dict()
        # with open(log_file_path, 'r', errors='ignore', encoding='utf-8') as record_file:
        #     train_info = json.load(fp=record_file)
        # with open(log_file_path, 'w', encoding='utf-8') as record_file:
        #     train_info[hyperparams] = iter
        #     json.dump(train_info, fp=record_file)
        time.sleep(16)
        self.epoches += 1
        return {"mean_accuracy": acc}


    def save_checkpoint(self, checkpoint_dir):
        print("++++++++++++++++" + checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        # torch.save(self.net.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        # self.net.load_state_dict(torch.load(checkpoint_path))
        pass
  

if __name__ == "__main__":
    args = parser.parse_args()
    total_gpus = args.gpus
    deadline = args.deadline
    ray.init(address=args.ray_address, num_cpus=total_gpus if args.smoke_test else None)

    hb = HyperBandScheduler()

    pbhs = PBHS(gpus_limit=total_gpus, metric_key="mean_accuracy", time_limit=deadline, max_iteration=200, exploration_ratio=1.0, dynamic_exploration = True, check_n_prediction_records=2)
    
    # pbhs = PBHS(metric_key="mean_accuracy", time_limit=7200, max_iteration=200, exploration_ratio=0.5)

    random.seed(0)
    weight_decay_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
    momentum_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.93, 0.96, 0.99, 0.997]
    # lr_list = [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 0.3, 0.7, 1e-5, 5e-5]
    # momentum_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.997, 0.93, 0.96, 0.98]
    # weight_decay_list = [0.0001, 0.1, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.7, 0.8, 0.9, 1, 0.001, 0.005, 0.0005, 0.01, 0.05, 1e-5, 5e-5]
    random.shuffle(weight_decay_list)
    random.shuffle(lr_list)
    random.shuffle(momentum_list)
    
    # lr_list = list()
    # momentum_list = list()
    # for i in range(0, 5):
    #     lr_list.append(random.uniform(0.0001, 0.1))
    
    # for i in range(0, 40):
    #     momentum_list.append(random.uniform(0.1, 0.99))
    
    

    analysis = tune.run(
        TrainCifarResNet,
        metric="mean_accuracy",
        mode="max",
        scheduler=pbhs,
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 3 if args.smoke_test else 200,
        },
        resources_per_trial={
            "cpu": 1,
            "gpu": 0,
        },
        # num_samples=1 if args.smoke_test else 3432,
        num_samples=936,
        checkpoint_at_end=False,
        checkpoint_freq=0,
        config={
            "args": args,
            "lr": tune.choice(lr_list),
            "weight_decay": tune.choice(weight_decay_list),
            "momentum": tune.choice(momentum_list),
        })

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))
