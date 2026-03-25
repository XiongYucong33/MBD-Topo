import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import setup_seed, MyDatasetShort, dgl_collate_func

# =============================
# 配置参数
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
n_splits = 5

# 五折模型保存根目录（和你的训练脚本保持一致）
save_root = "topo/chembl-bindingdb-human-KF/"

# 测试集路径（和你的训练脚本保持一致）
test_path = "data/chembl-and-drugbank-80-20/test.txt"

# 固定随机种子
setup_seed(0)


def find_model_path(save_root, fold):
    """
    兼容两种可能的保存方式：
    1. topo/chembl-bindingdb-human-KF/fold1/best.pth
    2. topo/chembl-bindingdb-human-KF/fold1best.pth
    """
    candidate_paths = [
        os.path.join(save_root, f"fold{fold}", "best.pth"),
        os.path.join(save_root, f"fold{fold}", "last.pth"),
        os.path.join(save_root, f"fold{fold}best.pth"),
        os.path.join(save_root, f"fold{fold}last.pth"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    return None


def evaluate_one_model(model, dataloader, device):
    """
    返回一折测试结果：
    acc, prc, recall, f1
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for smiles, sequences, g, labels in dataloader:
            g = g.to(device)
            labels = labels.long().to(device)

            outputs = model(smiles, sequences, g)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prc = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "acc": acc,
        "prc": prc,
        "recall": recall,
        "f1": f1
    }


def main():
    # =============================
    # 加载测试集
    # =============================
    test_dataset = MyDatasetShort(
        test_path,
        smile_length=138
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,   # 测试阶段不要丢最后一批
        collate_fn=dgl_collate_func
    )

    # =============================
    # 依次测试 5 折
    # =============================
    fold_results = []

    print("\n================ 五折测试开始 ================\n")

    for fold in range(1, n_splits + 1):
        model_path = find_model_path(save_root, fold)

        if model_path is None:
            print(f"[Fold {fold}] 未找到模型文件，跳过。")
            continue

        print(f"[Fold {fold}] 加载模型: {model_path}")

        # 这里因为你训练时保存的是整个 model 对象，所以直接 load 即可
        model = torch.load(model_path, map_location=device)
        model = model.to(device)

        result = evaluate_one_model(model, test_dataloader, device)
        fold_results.append(result)

        print(
            f"[Fold {fold}] "
            f"ACC={result['acc']:.4f} | "
            f"PRC={result['prc']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f}"
        )

    # =============================
    # 汇总平均值
    # =============================
    if len(fold_results) == 0:
        print("\n没有成功加载任何模型，无法计算平均值。")
        return

    avg_acc = np.mean([x["acc"] for x in fold_results])
    avg_prc = np.mean([x["prc"] for x in fold_results])
    avg_recall = np.mean([x["recall"] for x in fold_results])
    avg_f1 = np.mean([x["f1"] for x in fold_results])

    print("\n================ 五折测试结果汇总 ================\n")
    for i, result in enumerate(fold_results, start=1):
        print(
            f"Fold {i}: "
            f"ACC={result['acc']:.4f}, "
            f"PRC={result['prc']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}"
        )

    print("\n================ 平均值 ================\n")
    print(f"AVG_ACC    = {avg_acc:.4f}")
    print(f"AVG_PRC    = {avg_prc:.4f}")
    print(f"AVG_Recall = {avg_recall:.4f}")
    print(f"AVG_F1     = {avg_f1:.4f}")


if __name__ == "__main__":
    main()