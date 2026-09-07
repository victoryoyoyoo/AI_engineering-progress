"""
Lesson 8: 用 PyTorch optimizer 對照
"""
import torch


def demo_pytorch_optimizers():
    model = torch.nn.Linear(784, 10)

    sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)

    print("SGD (with momentum):", sgd)
    print("\nAdam:", adam)
    print("\nAdamW (decoupled weight decay, transformer常用):", adamw)
    print("\nScheduler:", scheduler)


if __name__ == "__main__":
    demo_pytorch_optimizers()
