#用面向对象的方式来处理文件路径
from pathlib import Path 

import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.utils.data import random_split

def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_transform, test_transform

def get_dataloaders(batch_size=128,num_workers=4,train_size=45000,val_size=5000):
    '''
    自动定位项目根目录 + 自动创建数据集文件夹，保证训练代码在任何环境都能直接运行
    '''
    #Path(_file)当前文件路径，.resolve()变成绝对路径
    #parents[0]上一级，parents[1]上上一级,即vit_cifar10
    #然后拼接得到root=vit_cifar10/data/CIFAR10
    root=Path(__file__).resolve().parents[1] /"data"/"CIFAR10"
    #如果这个文件不存在，就自动创建
    #parents=True,父级目录不存在自动创建,exists_ok=True文件夹已经存在也不报错
    root.mkdir(parents=True,exist_ok=True) 
    train_transform,test_transform=get_transforms()
    pin_memory=torch.cuda.is_available()
    full_train_dataset=datasets.CIFAR10(
        root=str(root), #pathlib.path转str
        train=True,
        transform=train_transform,
        download=False,
    )
    test_dataset=datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=test_transform,
        download=False
    )
    train_dataset,val_dataset=random_split(full_train_dataset,[train_size,val_size])


    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        drop_last=True
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        drop_last=False
    )
    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        drop_last=False
    )
    return train_loader,val_loader,test_loader
