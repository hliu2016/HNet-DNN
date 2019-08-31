import argparse
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os


parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')

folder = './PREDICT'
X_dim = 906 * 2

resultFolder = os.path.join(folder, 'breastCancer')
if not os.path.exists(resultFolder):
    print('mkdir {}'.format(resultFolder))
    os.mkdir(resultFolder)


drug_f = dict()
with open(os.path.join(folder, 'drug_feature.txt'), 'r') as f:
    for line in f:
        drug_id, fp = line.strip().split()
        fp = np.array(fp.split(','), dtype='float32')
        drug_f[drug_id] = fp

disease_f = dict()
with open(os.path.join(folder, 'disease_feature.txt'), 'r') as f:
    for line in f:
        disease_id, md = line.strip().split()
        md = np.array(md.split(','), dtype='float32')
        disease_f[disease_id] = md


class InteractionDataset(Dataset):
    def __init__(self, filename, root_dir=folder, transform=None):
        self.interaction = []
        with open(os.path.join(root_dir, filename), 'r') as f:
            for line in f:
                drug_id, disease_id = line.strip().split()
                self.interaction.append((drug_id, disease_id))


    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, idx):
        idx = idx % len(self)
        drug_id, disease_id = self.interaction[idx]

        drug_feature = drug_f[drug_id]
        disease_feature = disease_f[disease_id]

        X = np.concatenate([drug_feature, disease_feature])
        X = torch.from_numpy(X)
        return drug_id, disease_id, X


args = parser.parse_args()
cuda = torch.cuda.is_available()
if cuda:
    print('Using GPU')

seed = 10
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 2
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
epochs = args.epochs
weight_decay = 0.001
dropout = 0.2

params = {
    'n_classes': n_classes,
    'X_dim': X_dim,
    'train_batch_size': train_batch_size,
    'valid_batch_size': valid_batch_size,
    'epochs': epochs,
    'cuda': cuda
}
print('params: {}'.format(params))


def load_data(data_path='../data/'):
    print('loading data')
    case_dataset = InteractionDataset('case_breastCancer.txt')
    case_loader = torch.utils.data.DataLoader(case_dataset, batch_size=valid_batch_size, shuffle=True)
    return case_loader


class FC_DNN(nn.Module):
    def __init__(self):
        super(FC_DNN, self).__init__()
        self.lin1 = nn.Linear(X_dim, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)
        self.cat = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin3(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin4(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        xcat = self.cat(x)
        return xcat


def load_model(filename):
    return torch.load(filename)


def get_result_from_model(Q, data_loader):
    Q.eval()
    drugs = []
    diseases = []
    scores = []

    for batch_idx, (drug_id, disease_id, X) in enumerate(data_loader):
        X = X.view(-1, X_dim)
        X = Variable(X)
        if cuda:
            X = X.cuda()

        drugs.extend(drug_id)
        diseases.extend(disease_id)

        output = Q(X)
        output_probability = F.softmax(output, dim=1)
        if cuda:
            scores.extend(output_probability.cpu().data.numpy()[:, 1].tolist())
        else:
            scores.extend(output_probability.data.numpy()[:, 1].tolist())

    return drugs, diseases, scores


def write_result(probas1):
    with open(os.path.join(resultFolder, 'probas1-HNet-DNN.txt'), 'w') as f:
        for drug_id, disease_id, score in probas1:
            f.write('{} {} {}\n'.format(drug_id, disease_id, score))
        # for score in probas1:
        #     f.write('{}\n'.format(score))


if __name__ == '__main__':
    case_loader = load_data()

    best_model = 'best_DNN_model.pkl'
    fc_dnn = load_model(os.path.join(folder, best_model))

    drugs, diseases, scores = get_result_from_model(fc_dnn, case_loader)

    probas1 = []
    for i in range(len(drugs)):
        probas1.append((drugs[i], diseases[i], scores[i]))

    probas1.sort(key=lambda x: x[2], reverse=True)

    write_result(probas1)
    print('Done')
