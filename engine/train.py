
import json
import yaml

#控制python运行系统的工具箱
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
'''
python import的真实流程,step1:去sys.path里逐个找，在每个路径下找模块，找到就导入，找不到就报错
sys.path它就是一个python list,sys.path=["路径1","路径2",...,"路径n"]
'''
#D:\AI\projects\test_ai\cv_llm_learning\vit_cifar10
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0,str(root))
from datasets.data import get_dataloaders
from models.vit_tiny import Vit_Tiny

def evaluate_model(model,data_loader,device,criterion):
    model.eval()
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in data_loader:
            images,labels =images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            total_loss+=loss.item()*images.size(0)
            preds=outputs.argmax(1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    return total_loss/total,correct/total

def train_model(epochs=50,batch_size=128,lr=1e-3,device=None,save_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader,_=get_dataloaders(batch_size=batch_size)
    model=Vit_Tiny().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(model.parameters(),lr=lr)
    save_dir=Path(save_dir or root/"outputs") #Path的返回值是pathlib.Path,Path支持/拼接
    save_dir.mkdir(parents=True,exist_ok=True) #创建文件夹，如果父目录不存在也一起创建,如果目录已经存在不报错
    history=[]
    #记录四个参数train_loss,train_acc,val_loss,val_acc
    #train_loss:模型在训练集上的平均误差
    #train_acc:训练集上的准确率
    #val_loss:模型在验证集上的平均误差
    #val_acc:模型在验证集上的准确率
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        correct=0
        total=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*images.size(0) #加当前batch的总loss
            preds=outputs.argmax(1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
        train_loss=running_loss/total
        train_acc=correct/total
        val_loss,val_acc=evaluate_model(model,val_loader,device,criterion) #验证集目前是从测试集的数据，建议是从训练集划分一部分出来当验证集
        history.append({
            "epoch":epoch+1,
            "train_loss":float(train_loss),
            "train_acc":float(train_acc),
            "val_loss":float(val_loss),
            "val_acc":float(val_acc),
        })
        
        print(f"Epoch{epoch+1}:train_loss={train_loss:.4f},train_acc={train_acc:.4f}"
              f"val_loss={val_loss:.4f},val_acc={val_acc:.4f}")
        
        #把PyTorch的对象，模型/参数/字典保存到磁盘,torch.save(obj,path)
        #类型是dict(),model.state_dict()保存的是模型的所有可训练参数
        '''
        {
            "patch_embed.proj.weight": tensor(...),
            "patch_embed.proj.bias": tensor(...),

            "blocks.0.attn.q.weight": tensor(...),
            "blocks.0.attn.k.weight": tensor(...),

                "mlp.fc1.weight": tensor(...),
                ...
        }
        '''

        #1.checkpoint.pt（完整训练状态）
        # 把当前训练状态（模型+优化器+训练信息)保存到文件，方便以后继续训练或者恢复
        torch.save({
            "epoch":epoch+1,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "history":history,
        },save_dir/"checkpoint.pt")
        
        #2.history.json(训练日志)
        # 把训练模型的history以JSON文件形式保存到磁盘
        '''
        注意:JSON格式只能保存纯python可序列化类型，不能直接保存Tensor
        '''
        with open(save_dir/"history.json","w",encoding="utf-8") as f:
            json.dump(history,f,indent=2)
        
        #3.checkpoint.yaml(当前epoch元数据)
        # "把当前训练的关键指标+配置信息",保存成yaml文件，用于记录实验状态和复现实验
        with open(save_dir/"checkpoint.yaml","w",encoding="utf-8") as f:
            yaml.safe_dump({
                "epoch": epoch+1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(lr),
                "batch_size": int(batch_size),
            },f,sort_keys=False) #yaml.safe_dump(),把python字典写入.yaml文件,sort_keys=False保持字典写入时的字段顺序（提高可读性）
    
    #4.可选：保存vit_tiny_cifar10.pt(最终权重模型)
    torch.save({
        "model":model.state_dict(),
        "history":history,
    },save_dir/"vit_tiny_cifar10.pt")

    #5.history.yaml(训练日志)
    with open(save_dir/"history.yaml","w",encoding="utf-8") as f:
        yaml.safe_dump({"history":history},f,sort_keys=False)
    return history

if __name__=="__main__":
    train_model(epochs=50)
        
            
