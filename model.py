import torch.nn.functional as F
import torch.nn as nn
import torch
from dgllife.model.gnn import GCN


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, dropout=0.1):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(dim_in, dim_hid)
        self.fc_2 = nn.Linear(dim_hid, dim_out)

        self.ln = nn.LayerNorm(dim_out)
        self.dropout = dropout
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, X):
        X = F.relu(self.fc_1(X))
        X = F.dropout(X, self.dropout)
        X = F.relu(self.fc_2(X))
        return self.ln(X)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return self.softmax(x)


class DTI(nn.Module):
    def __init__(self, SMILES_length=290,
                 sequence_length=1700,
                 SMILES_in_feature=512,
                 sequence_in_feature=480,
                 mlp_hid_feature=256,
                 half_out_feature=128):
        super(DTI, self).__init__()

        self.mix_attention_layer = nn.MultiheadAttention(half_out_feature, 4)
        self.Drug_max_pool = nn.MaxPool1d(SMILES_length)
        self.Protein_max_pool = nn.MaxPool1d(sequence_length)
        self.dropout1 = nn.Dropout(0.1)
        self.drug_mlp = MLP(SMILES_in_feature, mlp_hid_feature,
                            half_out_feature, dropout=0.1)
        self.protein_mlp = MLP(sequence_in_feature, mlp_hid_feature,
                               half_out_feature, dropout=0.1)

    def forward(self, drug_features, protein_features):
        # print("0")
        # print(drug_features.shape, protein_features.shape)
        drug_features = self.drug_mlp(drug_features)
        protein_features = self.protein_mlp(protein_features)
        # print("1")
        drugConv = drug_features.permute(0, 2, 1)
        proteinConv = protein_features.permute(0, 2, 1)
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)
        drug_att, _ = self.mix_attention_layer(
            drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(
            protein_QKV, drug_QKV, drug_QKV)
        # print("2")
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        # result = torch.cat((drug_att, protein_att), dim=-1)
        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)

        return pair


class MYMOD(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(MYMOD, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.sequence_model = sequence_model.to(device)
        self.mynn1 = DTI().to(device)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''
        self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 128]).to(device)
        '''self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 17, padding=8),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 33, padding=16),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)'''
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.mynn2 = DTI(SMILES_in_feature=128,
                         sequence_in_feature=128).to(device)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=290).to(self.device)
            )['last_hidden_state']
            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700).to(self.device)
            )['last_hidden_state']
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)

        graph_features = self.gcn(G, G.ndata['h']).view(-1, 290, 128)
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float()).permute(0, 2, 1)

        pair1 = self.mynn1(smiles_features, sequence_features)
        pair2 = self.mynn2(graph_features, cnn_features)
        return self.mlp_classifier(torch.mul(pair1, pair2))


class NO_GRAPH_AND_CNN(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(NO_GRAPH_AND_CNN, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.sequence_model = sequence_model.to(device)
        self.mynn1 = DTI().to(device)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=290).to(self.device)
            )['last_hidden_state']
            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700).to(self.device)
            )['last_hidden_state']

        pair1 = self.mynn1(smiles_features, sequence_features)
        return self.mlp_classifier(pair1)


class NO_BERT_AND_ESM(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(NO_BERT_AND_ESM, self).__init__()

        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''
        self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 128]).to(device)
        '''self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 17, padding=8),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 33, padding=16),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)'''
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.mynn2 = DTI(SMILES_in_feature=128,
                         sequence_in_feature=128).to(device)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)

        graph_features = self.gcn(G, G.ndata['h']).view(-1, 290, 128)
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float()).permute(0, 2, 1)

        pair2 = self.mynn2(graph_features, cnn_features)
        return self.mlp_classifier(pair2)

class NO_BERT_AND_CNN(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(NO_BERT_AND_CNN, self).__init__()

        self.sequence_model = sequence_model.to(device)
        self.mynn1 = DTI(SMILES_in_feature=128).to(device)

        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''
        self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 128]).to(device)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():

            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700).to(self.device)
            )['last_hidden_state']
        graph_features = self.gcn(G, G.ndata['h']).view(-1, 290, 128)
        pair1 = self.mynn1(graph_features, sequence_features)
        return self.mlp_classifier(pair1)
    
class NO_GRAPH_AND_ESM(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(NO_GRAPH_AND_ESM, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.mynn1 = DTI(sequence_in_feature=128).to(device)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=290).to(self.device)
            )['last_hidden_state']
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float()).permute(0, 2, 1)
        pair1 = self.mynn1(smiles_features, cnn_features)
        return self.mlp_classifier(pair1)

    
class NO_DTI(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(NO_DTI, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.sequence_model = sequence_model.to(device)
        self.Drug_max_pool = nn.MaxPool1d(290)
        self.Protein_max_pool = nn.MaxPool1d(1700)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''
        self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 128]).to(device)
        '''self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 17, padding=8),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 33, padding=16),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)'''
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.fc_1 = nn.Linear(512+480, 256)
        self.fc_2 = nn.Linear(128+128, 256)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=290).to(self.device)
            )['last_hidden_state'].permute(0, 2, 1)
            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700).to(self.device)
            )['last_hidden_state'].permute(0, 2, 1)
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)

        graph_features = self.gcn(G, G.ndata['h']).view(-1, 290, 128).permute(0, 2, 1)
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float())
        pair1 = torch.cat([self.Drug_max_pool(smiles_features).squeeze(2),
                          self.Protein_max_pool(sequence_features).squeeze(2)], dim=1)
        pair2 = torch.cat([self.Drug_max_pool(graph_features).squeeze(2),
                          self.Protein_max_pool(cnn_features).squeeze(2)], dim=1)
        return self.mlp_classifier(torch.mul(self.fc_1(pair1),
                                             self.fc_2(pair2)))
    
    
class MYMOD2(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(MYMOD2, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.sequence_model = sequence_model.to(device)
        self.mynn1 = DTI(SMILES_in_feature=128).to(device)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 128]).to(device)
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.mynn2 = DTI(sequence_in_feature=128).to(device)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=290).to(self.device)
            )['last_hidden_state']
            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700).to(self.device)
            )['last_hidden_state']
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)

        graph_features = self.gcn(G, G.ndata['h']).view(-1, 290, 128)
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float()).permute(0, 2, 1)

        pair1 = self.mynn1(graph_features, sequence_features)
        pair2 = self.mynn2(smiles_features, cnn_features)
        return self.mlp_classifier(torch.mul(pair1, pair2))

class NO_GCN(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(MYMOD, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.sequence_model = sequence_model.to(device)
        self.mynn1 = DTI().to(device)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.mynn2 = DTI(SMILES_in_feature=512,
                         sequence_in_feature=128).to(device)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=290).to(self.device)
            )['last_hidden_state']
            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700).to(self.device)
            )['last_hidden_state']
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1700
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)

        graph_features = smiles_features
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float()).permute(0, 2, 1)

        pair1 = self.mynn1(smiles_features, sequence_features)
        pair2 = self.mynn2(graph_features, cnn_features)
        return self.mlp_classifier(torch.mul(pair1, pair2))
    
    
class MYMOD_SHORT(nn.Module):
    def __init__(self,  smiles_tokenrizer, smiles_model,
                 sequence_tokenrizer, sequence_model, device="cuda"):
        super(MYMOD_SHORT, self).__init__()
        self.smiles_model = smiles_model.to(device)
        self.sequence_model = sequence_model.to(device)
        self.mynn1 = DTI(SMILES_length=138, sequence_length=1000).to(device)
        self.smiles_tokenrizer = smiles_tokenrizer
        self.sequence_tokenrizer = sequence_tokenrizer
        self.device = device
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2).to(device)
        '''self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 512, 256, 128]).to(device)'''
        self.gcn = GCN(in_feats=75, hidden_feats=[
                       128, 256, 128]).to(device)
        '''self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 17, padding=8),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 33, padding=16),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)'''
        self.cnn = nn.Sequential(nn.Conv1d(33, 64, 3, padding=1),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Conv1d(64, 96, 5, padding=2),
                                 nn.BatchNorm1d(96),
                                 nn.ReLU(),
                                 nn.Conv1d(96, 128, 9, padding=4),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU()).to(device)
        self.mynn2 = DTI(SMILES_in_feature=128,
                         sequence_in_feature=128,
                         SMILES_length=138,
                         sequence_length=1000).to(device)

    def forward(self, SMILES, sequences, G):
        with torch.no_grad():
            smiles_features = self.smiles_model(**self.smiles_tokenrizer(
                SMILES,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=138).to(self.device)
            )['last_hidden_state']
            sequence_features = self.sequence_model(**self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1000).to(self.device)
            )['last_hidden_state']
            embeded_sequences = self.sequence_tokenrizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=1000
            )['input_ids'].to(self.device)
        embeded_sequences = F.one_hot(embeded_sequences, num_classes=33)

        graph_features = self.gcn(G, G.ndata['h']).view(-1, 138, 128)
        cnn_features = self.cnn(embeded_sequences.permute(
            0, 2, 1).float()).permute(0, 2, 1)

        pair1 = self.mynn1(smiles_features, sequence_features)
        pair2 = self.mynn2(graph_features, cnn_features)
        return self.mlp_classifier(torch.mul(pair1, pair2))