import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from data_list import ImageList
import os
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau
import xlrd
from scipy.optimize import curve_fit

import pandas as pd

import network.feature_extraction as feature_extraction
import network.feature_mapping as feature_mapping
import network.regression as feature_regression
import network.adnet as feature_adnet


def read_xlrd(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    dataFile = []
    for rowNum in range(table.nrows):
        if rowNum > 0:
            dataFile.append(table.row_values(rowNum))
    dataFile = sorted(dataFile)
    return dataFile


def cal_SROCC(pred, target):
    _, _, pred = logistic_5_fitting_no_constraint(pred, target)
    plcc, _ = pearsonr(pred, target)
    srocc, _ = spearmanr(pred, target)
    krocc, _ = kendalltau(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return plcc, srocc, krocc, rmse


def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted

def _save_checkpoint(epoch, model, checkpoint_dir='checkpoints/',
                     filename='model_'):
    filenamenew = os.path.join(checkpoint_dir, f'{filename + str(epoch)}.pth')
    torch.save({
            'model_0_state_dict': model[0].state_dict(),
            'model_1_state_dict': model[1].state_dict(),
            'model_2_state_dict': model[2].state_dict(),
            'model_3_state_dict': model[3].state_dict(),
            }, filenamenew)

def train(args, train_loader, train_loader1, optimizer, epoch, model):
    model[0].train()
    model[1].train()
    model[2].train()
    model[3].train()

    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target

    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source,_ = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target,_ = iter_target.next()
        data_target, label_target = data_target.cuda(), label_target.cuda()


        # Label scale normalization to 0-1
        label_source = (label_source) / args.slabelscale
        label_target = (label_target) / args.slabelscale

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()
        optimizer[3].zero_grad()

        feature1 = model[0](torch.cat((data_source, data_target), 0))
        feature2 = model[1](feature1)
        score1 = model[2](feature1)
        score = model[2](feature2)



        srocc_latter, _ = spearmanr(score.narrow(0, 0, data_source.size(0))[:, 0].detach().cpu(),
                                    label_source.detach().cpu())
        srocc_former, _ = spearmanr(score1.narrow(0, 0, data_source.size(0))[:, 0].detach().cpu(),
                                    label_source.detach().cpu())
        if srocc_latter > srocc_former + 0.1:
            Dfake_source = 1
        else:
            Dfake_source = 0

        Dfake_target = 0

        loss3 = model[3](feature2, score, Dfake_source, Dfake_target, data_source.size(0), data_target.size(0))

        loss1 = F.mse_loss(score.narrow(0, 0, data_source.size(0))[:,0], label_source.float())

        loss = loss1 + loss3

        loss.backward()
        optimizer[0].step()
        optimizer[1].step()
        optimizer[2].step()
        optimizer[3].step()

        if (batch_idx + epoch * num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, num_iter * args.batch_size,
                       100. * batch_idx / num_iter, loss.item(), loss1.item(), loss3.item()))
    _save_checkpoint(epoch, model)


def test(args, model, test_loader, epoch):
    model[0].eval()
    model[1].eval()
    model[2].eval()
    model[3].eval()

    pred_all = np.array([])
    target_all = np.array([])
    path_all = np.array([])

    for data, target, path in test_loader:
        data, target = data.cuda(), target.cuda()

        # Label scale normalization to 0-1
        target = (target) / args.tlabelscale

        feature1 = model[0](data)
        feature2 = model[1](feature1)
        output = model[2](feature2)
        pred = output.data.cpu().view_as(target).numpy()
        target = target.data.cpu().numpy()
        pred_all = np.concatenate((pred_all, pred), axis=0)
        target_all = np.concatenate((target_all, target), axis=0)
        path_all = np.concatenate((path_all, path), axis=0)

    plcc, srocc, krocc, rmse = cal_SROCC(pred_all, target_all)

    path_all = path_all.reshape(-1, 1)
    target_all = target_all.reshape(-1, 1)
    pred_all = pred_all.reshape(-1, 1)
    all_results = np.concatenate((path_all, target_all, pred_all), axis=1)
    results2 = pd.DataFrame(columns=['plyname', 'MOS', 'pred'], data=all_results)
    results2.to_csv(f'results/test_pre_score{str(epoch)}.csv',index=False)

    print(' '.join([
        f"PLCC: {plcc:.6f},",
        f"SROCC: {srocc:.6f}, ",
        f"KROCC: {krocc:.6f}, ",
        f"RMSE: {rmse:.6f}, "
    ]))

    return plcc, srocc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='IT-PCQA')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--slabelscale', type=float, default=9.0,
                        help='Maximum value of labels for source domain')
    parser.add_argument('--tlabelscale', type=float, default=9.0,
                        help='Maximum value of labels for target domain')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
    parser.add_argument('--resume', type=str, default=None,
                  help='path for loading the checkpoint')
    parser.add_argument('--backbone', type=str, default='HSCNN',
                        help='backbone')

    args = parser.parse_args()

    # random
    if not args.random:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # loading dataset (source domain for images, and target domain for point cloud projections)
    source_list = 'config/TID2013/mos_with_names.txt'
    target_list = 'config/SJTU-PCQA/label_yq0-9_train.txt'
    test_list = 'config/SJTU-PCQA/label_yq0-9_val.txt'

    # resize
    backbone = args.backbone
    if backbone == 'HSCNN':
        pic_resize = 224
        channel = 256

    # create DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
            transforms.Resize((pic_resize, pic_resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='RGB'),
        batch_size=args.batch_size, num_workers=1, shuffle=True)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
            transforms.Resize((pic_resize, pic_resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='RGB'),
        batch_size=args.batch_size, num_workers=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
            transforms.Resize((pic_resize, pic_resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='RGB'),
        batch_size=args.test_batch_size, num_workers=1, shuffle=True)

    extraction = feature_extraction.Encoder(backbone = backbone)
    extraction = extraction.cuda()
    mapping = feature_mapping.Feature_mapping(channel, channel)
    mapping = mapping.cuda()
    regression = feature_regression.Regression(channel)
    regression = regression.cuda()
    adnet = feature_adnet.AdversarialNetwork(channel)
    adnet = adnet.cuda()
    model = [extraction, mapping, regression, adnet]

    if args.resume:
        checkpoint = torch.load(args.resume)
        model[0].load_state_dict(checkpoint['model_0_state_dict'], strict=True)
        model[1].load_state_dict(checkpoint['model_1_state_dict'], strict=True)
        model[2].load_state_dict(checkpoint['model_2_state_dict'], strict=True)
        model[3].load_state_dict(checkpoint['model_3_state_dict'], strict=True)

    # SGD optimizer
    optimizer_model = optim.SGD(model[0].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)
    optimizer_mapping = optim.SGD(model[1].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)
    optimizer_regression = optim.SGD(model[2].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)
    optimizer_adnet = optim.SGD(model[3].parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)

    optimizer = [optimizer_model, optimizer_mapping, optimizer_regression, optimizer_adnet]

    result = []
    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, train_loader1, optimizer, epoch, model)
        plcc, srocc = test(args, model, test_loader, epoch)
        result.append([plcc, srocc])
        resultlist = pd.DataFrame(columns=['plcc', 'SROCC'], data=result)
        resultlist.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
