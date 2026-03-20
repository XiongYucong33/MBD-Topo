from torch.utils.data import Dataset
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import smiles_to_bigraph
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import torch
import dgl


def setup_seed(seed=0):
    import torch

    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("set cpu seed")
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print("set cuda seed")


class MyDataset(Dataset):
    def __init__(self, path, cut=None, default_smile_length=290):

        self.node_featurizer1 = CanonicalAtomFeaturizer()
        self.edge_featurizer1 = CanonicalBondFeaturizer(self_loop=True)
        with open(path, "r") as f:
            lines = f.readlines()
        if cut is not None:
            lines = lines[:cut]
        # random.shuffle(lines)
        self.lines = [line.split() for line in lines]
        new_lines=[]
        for line in self.lines:
            try:
                line[2] = float(line[2])

                # node_featurizer1 = CanonicalAtomFeaturizer()
                # edge_featurizer1 = CanonicalBondFeaturizer(self_loop = True)
                g = smiles_to_bigraph(smiles=line[0],
                                      node_featurizer=self.node_featurizer1,
                                      edge_featurizer=self.edge_featurizer1,
                                      add_self_loop=True)
                # init_transform = nn.Linear(75, 128, bias=False)
                actual_node_feats = g.ndata.pop('h')
                num_actual_nodes = actual_node_feats.shape[0]
                num_virtual_nodes = default_smile_length - num_actual_nodes
                virtual_node_bit = torch.zeros([num_actual_nodes, 1])
                virtual_node_bit
                actual_node_feats = torch.cat(
                    (actual_node_feats, virtual_node_bit), 1)

                g.ndata['h'] = actual_node_feats
                virtual_node_feat = torch.cat(
                    (torch.zeros(num_virtual_nodes, 74),
                     torch.ones(num_virtual_nodes, 1)), 1)

                g.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
                g = g.add_self_loop()
                # virtual_node_feat.shape
                line.append(g)
                new_lines.append(line)
            except:
                print("!")
                
            self.lines=new_lines
        print(f"loaded  {len(self.lines)}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def dgl_collate_func(lines):
    smile_list = []
    sequence_list = []
    label_list = []
    graph_list = []
    for smiles, sequence, label, graph in lines:
        smile_list.append(smiles)
        sequence_list.append(sequence)
        label_list.append(label)
        graph_list.append(graph)

    batched_graphs = dgl.batch(graph_list)
    batched_labels = torch.tensor(label_list)
    # x_remain = [list(i[1:]) for i in x]
    return smile_list, sequence_list, batched_graphs, batched_labels


def train_an_epoch(model, train_dataloader, eval_dataloader, optimizer, loss,
                   best_eval_loss, device, save_model, save_path):

    model.train()
    loss_all = 0
    loss_count = 0

    for smiles, sequences, g, labels in train_dataloader:
        # print("GGGGG"+g)
        y_hat = model(smiles, sequences, g.to(device))
        # print(y_hat)
        # print(labels)
        ls = loss(y_hat, labels.long().to(device))
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        """
        y2 = torch.argmax(y_hat, axis=1)

        num_good = (y2 == labels.long().to(device)).sum().item()
        num_all = y2.shape[0]
        acc = num_good/num_all
        """
        # print(f"loss:  {ls:f}  acc:  {acc:f}")
        loss_all += ls
        loss_count += 1
    loss_all = loss_all/loss_count
    print(f"train loss:  {loss_all:f}")
    print("*******************eval**************************")
    model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    if eval_dataloader is not None:
        with torch.no_grad():

            loss_all = 0
            loss_count = 0
            for smiles, sequences, g, labels in eval_dataloader:

                y_hat = model(smiles,
                              sequences, g.to(device))
                ls = loss(y_hat, labels.long().to(device))
                # y2 = np.argmax(y_hat.to('cpu').detach().numpy(), axis=1)

                # for i in range(y2.shape[0]):
                #    if y2[i] == labels[i]:
                #        num_good += 1
                # num_all += y2.shape[0]
                y2 = torch.argmax(y_hat, axis=1)
                gt = labels.long().to(device)
                for i in range(len(y2)):
                    if y2[i] == 0 and gt[i] == 0:
                        tn += 1
                    if y2[i] == 1 and gt[i] == 0:
                        fp += 1
                    if y2[i] == 0 and gt[i] == 1:
                        fn += 1
                    if y2[i] == 1 and gt[i] == 1:
                        tp += 1

                loss_all += ls
                loss_count += 1
            loss_all = loss_all/loss_count
            acc = (tn+tp)/(tn+tp+fn+fp)
            print(f"tp {tp} ,tn {tn} ,fp {fp} ,fn {fn}")
            print(f"eval loss:  {loss_all:f}    ,acc:   {acc}")
            if save_model and loss_all < best_eval_loss:
                best_eval_loss = loss_all
                torch.save(model, save_path+'best.pth')
                # print("*******************eval**************************")
                print(" "*60+"save")
    return best_eval_loss


def test(model, test_dataloader, loss, device):

    model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_list = []
    score_list = []
    with torch.no_grad():

        loss_all = 0
        loss_count = 0
        for smiles, sequences, g, labels in test_dataloader:

            y_hat = model(smiles,
                          sequences, g.to(device))
            ls = loss(y_hat, labels.long().to(device))
            # y2 = np.argmax(y_hat.to('cpu').detach().numpy(), axis=1)

            # for i in range(y2.shape[0]):
            #    if y2[i] == labels[i]:
            #        num_good += 1
            # num_all += y2.shape[0]
            y2 = torch.argmax(y_hat, axis=1)
            gt = labels.long().to(device)
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
            loss_all += ls
            loss_count += 1
        loss_all = loss_all/loss_count
        if tp+fp==0:
            precision=0
        else:
            precision = tp/(tp+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)
        if tp+fn==0:
            recall=0
        else:
            recall = tp/(tp+fn)
        fpr, tpr, _ = roc_curve(true_list, score_list)
        pr, re, _ = precision_recall_curve(
            true_list, score_list, pos_label=1)
        if precision+recall==0:
            f1=0
        else:
            f1 = 2*recall*precision/(precision+recall)
        auroc = auc(fpr, tpr)
        auprc = auc(re, pr)
        # roc_auc = roc_auc_score(true_list, score_list)
        print("*******************test**************************")
        print(f"tp {tp} ,tn {tn} ,fp {fp} ,fn {fn}")
        # print(f"test loss:  {loss_all:f}    ,acc:   {acc}")
        # print(recall.dtype, precision.dtype, f1.dtype, auroc.dtype)
        # print(acc.dtype)
        print(f"acc {acc:f} ,recall {recall:f} ,precision {precision:f}")
        print(
            f"f1 {f1:f} ,auroc {auroc:f} ,auprc {auprc:f}")
        print("**************************************************\n\n")


def test_out(model, test_dataloader,  device, path):

    model.eval()
    true_list = []
    score_list = []

    with torch.no_grad():

        for smiles, sequences, g, labels in test_dataloader:

            y_hat = model(smiles,
                          sequences, g.to(device))

            # y2 = np.argmax(y_hat.to('cpu').detach().numpy(), axis=1)

            # for i in range(y2.shape[0]):
            #    if y2[i] == labels[i]:
            #        num_good += 1
            # num_all += y2.shape[0]

            gt = labels.long().to(device)
            for i in range(len(y_hat)):
                true_list.append(gt[i].to("cpu"))
                score_list.append(y_hat[i][1].to("cpu"))
    with open(path, "w") as f:
        for item1, item2 in zip(true_list, score_list):
            f.write(f"{item1} {item2}\n")
            
            
def train(model, train_dataset, val_dataset,  optimizer,
          loss,  epoches,  best_eval_loss, device,
          save_model, save_path, test_dataset=None, scheduler=None):
    # if test_lines is not None:
    #   test(model, test_lines, loss, batch_size, device)
    for i in range(epoches):
        print(f"epoch\t{i}")
        best_eval_loss = train_an_epoch(model, train_dataset,
                                        val_dataset,
                                        optimizer, loss,
                                        best_eval_loss, device,
                                        save_model, save_path)
        if scheduler is not None:
            scheduler.step()
        if test_dataset is not None:
            #    test(model, test_dataloader, loss, device)
            test(model, test_dataset, loss, device)
    if save_model:
        torch.save(model, save_path+'last.pth')

        
class MyDatasetShort(Dataset):
    def __init__(self, path, cut=None, smile_length=290, sequence_length=1000):

        self.node_featurizer1 = CanonicalAtomFeaturizer()
        self.edge_featurizer1 = CanonicalBondFeaturizer(self_loop=True)
        with open(path, "r") as f:
            lines = f.readlines()
        if cut is not None:
            lines = lines[:cut]
        # random.shuffle(lines)
        self.lines = [line.split() for line in lines]
        self.lines = [line for line in self.lines if len(
            line[0]) < smile_length]
        # c = 0
        for i,line in enumerate(self.lines):
            line[2] = float(line[2])
            # print(f"line :{i} {line[0]}")
            if (len(line[1]) > sequence_length):
                half_length = sequence_length // 2
                line[1] = line[1][:half_length] + line[1][-half_length:]
            # c += 1
            # print(c)
            # print(f"smiles: {line[0]}")
            # node_featurizer1 = CanonicalAtomFeaturizer()
            # edge_featurizer1 = CanonicalBondFeaturizer(self_loop = True)
            g = smiles_to_bigraph(smiles=line[0],
                                  node_featurizer=self.node_featurizer1,
                                  edge_featurizer=self.edge_featurizer1,
                                  add_self_loop=True)
            # init_transform = nn.Linear(75, 128, bias=False)
            actual_node_feats = g.ndata.pop('h')
            num_actual_nodes = actual_node_feats.shape[0]
            num_virtual_nodes = smile_length - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            virtual_node_bit
            actual_node_feats = torch.cat(
                (actual_node_feats, virtual_node_bit), 1)

            g.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat(
                (torch.zeros(num_virtual_nodes, 74),
                 torch.ones(num_virtual_nodes, 1)), 1)

            g.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            g = g.add_self_loop()
            # virtual_node_feat.shape
            line.append(g)
        print("loaded")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]