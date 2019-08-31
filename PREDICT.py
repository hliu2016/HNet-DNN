import argparse
import time
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import os
from sklearn import metrics
import copy
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc


parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')

folder = './PREDICT'
X_dim = 906 * 2

resultFolder = os.path.join(folder, 'result')
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
        self.label = []
        with open(os.path.join(root_dir, filename), 'r') as f:
            for line in f:
                drug_id, disease_id, label = line.strip().split()
                self.interaction.append((drug_id, disease_id))
                self.label.append(int(label))

        self.label = torch.LongTensor(self.label)

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, idx):
        idx = idx % len(self)
        drug_id, disease_id = self.interaction[idx]

        drug_feature = drug_f[drug_id]
        disease_feature = disease_f[disease_id]

        X = np.concatenate([drug_feature, disease_feature])
        X = torch.from_numpy(X)
        label = self.label[idx]
        return X, label


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
    train_dataset = InteractionDataset('train.txt')
    valid_dataset = InteractionDataset('valid.txt')
    test_dataset = InteractionDataset('test.txt')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


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


def report_loss(epoch, loss):
    print()
    print('Epoch-{}; loss: {:.4}'.format(epoch, loss))


def save_model(model, filename):
    torch.save(model, filename)


def load_model(filename):
    return torch.load(filename)


def classification_accuracy(Q, data_loader):
    Q.eval()
    labels = []
    scores = []

    loss = 0
    correct = 0
    for batch_idx, (X, target) in enumerate(data_loader):
        X = X.view(-1, X_dim)
        X, target = Variable(X), Variable(target)

        if cuda:
            X, target = X.cuda(), target.cuda()

        output = Q(X)
        output_probability = F.softmax(output, dim=1)

        labels.extend(target.data.tolist())
        if cuda:
            scores.extend(output_probability.cpu().data.numpy()[:, 1].tolist())
        else:
            scores.extend(output_probability.data.numpy()[:, 1].tolist())

        loss += F.cross_entropy(output, target, size_average=False).data[0]

        pred = output_probability.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    loss /= len(data_loader)
    acc = correct / len(data_loader.dataset)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return loss, acc, auc


def train(fc_dnn, fc_solver, train_labeled_loader):
    TINY = 1e-15

    fc_dnn.train()
    for X, target in train_labeled_loader:
        X, target = Variable(X), Variable(target)
        if cuda:
            X = X.cuda()
            target = target.cuda()

        X = X.view(-1, X_dim)
        out = fc_dnn(X)

        fc_solver.zero_grad()
        loss = F.cross_entropy(out + TINY, target)
        loss.backward()
        fc_solver.step()

    return loss.data[0]


def generate_model(train_labeled_loader, valid_loader):
    print('generating new model')
    torch.manual_seed(10)
    model = FC_DNN()
    if cuda:
        model = model.cuda()

    lr = 0.001
    fc_solver = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start = time.time()
    max_valid_acc, max_valid_auc = 0, 0
    result_model = None

    train_loss, train_acc, train_auc = classification_accuracy(model, train_labeled_loader)
    print('no train loss {}, acc {}, auc {}'.format(train_loss, train_acc, train_auc))

    for epoch in range(epochs):
        loss = train(model, fc_solver, train_labeled_loader)
        report_loss(epoch, loss)

        train_loss, train_acc, train_auc = classification_accuracy(model, train_labeled_loader)
        print('Train loss {:.4}, acc {:.4}, auc {:.4}'.format(train_loss, train_acc, train_auc))

        valid_loss, valid_acc, valid_auc = classification_accuracy(model, valid_loader)
        if valid_acc > max_valid_acc:
            max_valid_acc, max_valid_auc = valid_acc, valid_auc
            result_model = copy.deepcopy(model)

        if valid_auc > 0.91:
            fc_solver = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    end = time.time()
    print('Training time: {:.4} seconds'.format(end - start))
    return result_model


def get_result_from_model(Q, data_loader):
    Q.eval()
    labels = []
    scores = []

    for batch_idx, (X, target) in enumerate(data_loader):
        X = X.view(-1, X_dim)
        X, target = Variable(X), Variable(target)  # target in [0, 9]

        if cuda:
            X, target = X.cuda(), target.cuda()

        output = Q(X)
        output_probability = F.softmax(output, dim=1)

        labels.extend(target.data.tolist())
        if cuda:
            scores.extend(output_probability.cpu().data.numpy()[:, 1].tolist())
        else:
            scores.extend(output_probability.data.numpy()[:, 1].tolist())

    return scores, labels


def write_result(probas1, y):
    with open(os.path.join(resultFolder, 'probas1-HNet-DNN.txt'), 'w') as f:
        for val in probas1:
            f.write('{}\n'.format(val))

    with open(os.path.join(resultFolder, 'y-HNet-DNN.txt'), 'w') as f:
        for val in y:
            f.write('{}\n'.format(val))


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = load_data()

    best_model = 'best_DNN_model.pkl'
    if not os.path.exists(os.path.join(folder, best_model)):
        fc_dnn = generate_model(train_loader, valid_loader)
        save_model(fc_dnn, os.path.join(folder, best_model))
    else:
        fc_dnn = load_model(os.path.join(folder, best_model))

    test_loss, test_acc, test_auc = classification_accuracy(fc_dnn, test_loader)
    print()
    print('Test loss {:.4}, acc {:.4}, auc {:.4}'.format(test_loss, test_acc, test_auc))

    probas1, y = get_result_from_model(fc_dnn, test_loader)
    average_precision = average_precision_score(y, probas1)
    print('Test aupr {:.4}'.format(average_precision))
    fpr, tpr, thresholds = roc_curve(y, probas1, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('Test auc {:.4}'.format(roc_auc))
    write_result(probas1, y)
    print('Done')
