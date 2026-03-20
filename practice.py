from utils import test, dgl_collate_func, MyDataset
# from model import *

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
# import configs
# from model import MY_Attentional_NN
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import random
import datetime
device = "cuda"


def practice(model, test_dataloader,  device, save_path):
    model = model.to(device)
    model.eval()
    datetime_object1 = datetime.datetime.now()
    with torch.no_grad():
        for smiles, sequences, g, labels in test_dataloader:

            y_hat = model(smiles, sequences, g.to(device))

            # print(y_hat)
            # gt = labels.long().to(device)
            for i in range(len(y_hat)):
                if y_hat[i][1] > 0.99999:
                    with open(save_path, "a") as f:
                        f.write(f"{smiles[i]} {y_hat[i][1]}")
                        f.write("\n")
                    print(y_hat[i][1], smiles[i])
            datetime_object2 = datetime.datetime.now()
            print(datetime_object2-datetime_object1)
            datetime_object1 = datetime_object2
            '''
            for i in range(len(y2)):
                if y2[i] == 0 and gt[i] == 0:
                    tn += 1
                if y2[i] == 1 and gt[i] == 0:
                    fp += 1
                if y2[i] == 0 and gt[i] == 1:
                    fn += 1
                if y2[i] == 1 and gt[i] == 1:
                    tp += 1
                true_list.append(gt[i].to("cpu"))
                score_list.append(y_hat[i][1].to("cpu"))
            '''
        # roc_auc = roc_auc_score(true_list, score_list)
        print("*******************test**************************")

        # print(f"test loss:  {loss_all:f}    ,acc:   {acc}")
        # print(recall.dtype, precision.dtype, f1.dtype, auroc.dtype)
        # print(acc.dtype)

        # print("**************************************************\n\n")


model_path = "topo/chembl-balance/best-best.pth"
mymod = torch.load(model_path)
test_dataset = MyDataset("data/p11388_bindingdb_2.txt")
save_path="topo/positive/P11388_bindingdb_top.txt"
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32,
                             shuffle=True, drop_last=False,
                             collate_fn=dgl_collate_func)
practice(mymod, test_dataloader,  device, save_path)