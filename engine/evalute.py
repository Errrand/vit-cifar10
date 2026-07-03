import sys
from pathlib import Path
import torch
import torch.nn as nn
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0,str(root))
from datasets.data import get_dataloaders
from engine.train import evaluate_model
from models.vit_tiny import Vit_Tiny

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Vit_Tiny().to(device)
    criterion=nn.CrossEntropyLoss()
    
    checkpoint=torch.load(root/"outputs"/"vit_tiny_cifar10.pt",map_location=device)
    model.load_state_dict(checkpoint["model"])
    _,test_loader=get_dataloaders(batch_size=128)
    _,test_acc=evaluate_model(model,test_loader,device,criterion)
    print(f"test_acc={test_acc:.4f}")
    
if __name__=="__main__":
    main()
