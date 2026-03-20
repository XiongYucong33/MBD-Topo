import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel

from model import DTI, MYMOD_SHORT
from utils import setup_seed, MyDatasetShort, train, dgl_collate_func

# -----------------------------
# 配置参数
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
load_online = False
n_splits = 5
batch_size = 16
num_epochs = 100
init_best_eval_loss = float("inf")  # 记录最优模型
save_root = "topo/chembl-bindingdb-human-KF/"

# 设置随机种子保证复现
setup_seed(0)

# -----------------------------
# 加载编码器 / tokenizer
# -----------------------------
if load_online:
    try:
        tokenizer1 = AutoTokenizer.from_pretrained("UdS-LSV/smole-bert")
        model1 = AutoModel.from_pretrained("UdS-LSV/smole-bert")
        tokenizer2 = AutoTokenizer.from_pretrained(
            "facebook/esm2_t12_35M_UR50D")
        model2 = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        print("Loaded encoders from HF Hub")
    except Exception:
        tokenizer1 = torch.load("tokenizer1.pth")
        model1 = torch.load("model1.pth")
        tokenizer2 = torch.load("tokenizer2.pth")
        model2 = torch.load("model2.pth")
        print("Loaded encoders from local .pth")
else:
    tokenizer1 = torch.load("tokenizer1.pth")
    model1 = torch.load("model1.pth")
    tokenizer2 = torch.load("tokenizer2.pth")
    model2 = torch.load("model2.pth")
    print("Loaded encoders from local .pth")

# -----------------------------
# 数据集
#   - 不再用固定 eval.txt
#   - 只用 train.txt 做 5 折划分
#   - test.txt 保持不变
# -----------------------------
full_dataset = MyDatasetShort('data/chembl-and-drugbank-80-20/train.txt', smile_length=138)
test_dataset = MyDatasetShort('data/chembl-and-drugbank-80-20/test.txt', smile_length=138)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True,
                             collate_fn=dgl_collate_func)

# -----------------------------
# 5 折交叉验证
# -----------------------------
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
indices = np.arange(len(full_dataset))

for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
    print(
        f"\n==================== Fold {fold}/{n_splits} ====================")

    # 每折的训练集与验证集
    train_subset = Subset(full_dataset, train_idx.tolist())
    val_subset = Subset(full_dataset, val_idx.tolist())

    train_dataloader = DataLoader(dataset=train_subset, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  collate_fn=dgl_collate_func)
    eval_dataloader = DataLoader(dataset=val_subset, batch_size=batch_size,
                                 shuffle=False, drop_last=True,
                                 collate_fn=dgl_collate_func)

    # 每折重新初始化模型
    mymod = MYMOD_SHORT(tokenizer1, model1, tokenizer2, model2).to(device)

    optimizer = torch.optim.Adam(mymod.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    # 保存路径
    save_path = os.path.join(save_root, f"fold{fold}")
    os.makedirs(save_path, exist_ok=True)

    # 训练并验证
    train(mymod, train_dataloader, eval_dataloader, optimizer,
          loss_fn, num_epochs, init_best_eval_loss, device,
          save_model=True, save_path=save_path,  # 改成 True 可以保存模型
          test_dataset=test_dataloader, scheduler=None)
