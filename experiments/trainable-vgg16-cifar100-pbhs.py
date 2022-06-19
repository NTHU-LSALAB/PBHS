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
# from ray.tune.schedulers.pbhs import PBHS
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.pbhs import PBHS
from ray.tune.schedulers.ashav2 import ASHAv2
import random
import argparse
from models import *
import time


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

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='/home/didwdidw/project/Cifar-100', train=True, download=True, transform=transform_train)
    cifar100_training_loader = torch.utils.data.DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='/home/didwdidw/project/Cifar-100', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std



class TrainCifarVGG(tune.Trainable):

    def setup(self, config):
        self.epoches = 0
        BATCH_SIZE = 32
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        # cudnn.benchmark = True
        self.device = torch.device("cuda")
            # 加载数据集
        # 标准化为范围在[-1, 1]之间的张量
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # # 数据集
        # # # 利用自定义 Dataset
        # # trainset = Cifar10Dataset(root='./Cifar-10', train=True, transform=transform)  # 训练数据集
        # # testset = Cifar10Dataset(root='./Cifar-10', train=False, transform=transform)
        # # 利用库函数进行数据集加载
        # print("Download trainset")
        # trainset = torchvision.datasets.CIFAR100(root='/home/didwdidw/project/Cifar-100', train=True, download=True, transform=transform_train)  # 训练数据集
        # print("Download testset")
        # testset = torchvision.datasets.CIFAR100(root='/home/didwdidw/project/Cifar-100', train=False, download=True, transform=transform_test)
        

        # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
        self.trainloader = get_training_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=2,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.testloader = get_test_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=2,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.net = VGG(vgg_name="VGG16", num_class=100).to(self.device)

        self.writer = SummaryWriter('./logs_resNet')

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题

        # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.get("lr", 0.1), momentum=config.get("momentum", 0.9), weight_decay=config.get("weight_decay", 5e-4))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def step(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        start_time = time.time()
        acc = self.Accuracy(self.testloader, self.net, self.device) / 100
        end_time = time.time()
        print(end_time - start_time)
        print('[%d, %5f] loss: %.3f' % (self.epoches + 1, acc, train_loss / 2000))
        self.scheduler.step()
        self.epoches += 1
        return {"mean_accuracy": acc}


    def save_checkpoint(self, checkpoint_dir):
        print("++++++++++++++++" + checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.net.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path))

    def Accuracy(self, testloader, net, device):
        self.net.eval()
        # 使用测试数据测试网络
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:

                images, labels = data
                # 将输入和目标在每一步都送入GPU
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        return 100.0 * correct / total

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(address=args.ray_address, num_cpus=32 if args.smoke_test else None)
    total_gpus = args.gpus
    max_iters = 250
    experiment_deadline = args.deadline
    # hb = HyperBandScheduler(max_t=200)
    # ahb = AsyncHyperBandScheduler(max_t=200)
    # pbhs = PBHS(metric_key="mean_accuracy", time_limit=21600, max_iteration=200, exploration_ratio=0.5)
    pbhs = PBHS(gpus_limit=total_gpus, metric_key="mean_accuracy", time_limit=experiment_deadline, max_iteration=max_iters, exploration_ratio=1.0, dynamic_exploration = True, check_n_prediction_records=2)
    
    random.seed(0)
    weight_decay_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
    momentum_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.93, 0.96, 0.99, 0.997]
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
        TrainCifarVGG,
        # metric="mean_accuracy",
        # mode="max",
        scheduler=pbhs,
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 3 if args.smoke_test else max_iters,
        },
        resources_per_trial={
            "cpu": 4,
            "gpu": 1,
        },
        checkpoint_at_end=False,
        checkpoint_freq=0,
        num_samples=936,
        config={
            "args": args,
            "lr": tune.choice(lr_list),
            "weight_decay": tune.choice(weight_decay_list),
            "momentum": tune.choice(momentum_list),
        })

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))