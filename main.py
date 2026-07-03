import sys
from pathlib import Path

root=Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0,str(root))

from engine.train import train_model
def main():
    print("Start training Vit-Tiny on CIFAR10")
    train_model(epochs=200,batch_size=128,lr=1e-3)

if __name__ == "__main__":
    main()
