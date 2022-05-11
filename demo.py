import argparse
import torch
from torchvision import transforms
from data_list import ImageList
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import xlrd
from scipy.optimize import curve_fit
import network.feature_extraction as feature_extraction
import network.feature_mapping as feature_mapping
import network.regression as feature_regression
import network.adnet as feature_adnet
from tqdm import tqdm


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

def test(args, model, test_loader):
    model[0].eval()
    model[1].eval()
    model[2].eval()
    model[3].eval()

    pred_all = np.array([])
    target_all = np.array([])
    path_all = np.array([])

    for data, target, path in tqdm(test_loader):
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
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='input batch size for testing')
    parser.add_argument('--slabelscale', type=float, default=9.0,
                        help='Maximum value of labels for source domain')
    parser.add_argument('--tlabelscale', type=float, default=9.0,
                        help='Maximum value of labels for target domain')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--resume', type=str, default='checkpoints/model_15.pth',
                  help='path for loading the checkpoint')

    args = parser.parse_args()

    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # loading dataset (source domain for images, and target domain for point cloud projections)
    test_list = 'config/SJTU-PCQA/label_yq0-9_val.txt'

    # resize
    pic_resize = 224
    channel = 256

    # create DataLoader
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
            transforms.Resize((pic_resize, pic_resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='RGB'),
        batch_size=args.test_batch_size, num_workers=1, shuffle=True)

    extraction = feature_extraction.Encoder()
    extraction = extraction.cuda()
    mapping = feature_mapping.Feature_mapping(channel, channel)
    mapping = mapping.cuda()
    regression = feature_regression.Regression(channel)
    regression = regression.cuda()
    adnet = feature_adnet.AdversarialNetwork(channel)
    adnet = adnet.cuda()
    model = [extraction, mapping, regression, adnet]

    checkpoint = torch.load(args.resume)
    model[0].load_state_dict(checkpoint['model_0_state_dict'], strict=True)
    model[1].load_state_dict(checkpoint['model_1_state_dict'], strict=True)
    model[2].load_state_dict(checkpoint['model_2_state_dict'], strict=True)
    model[3].load_state_dict(checkpoint['model_3_state_dict'], strict=True)

    plcc, srocc = test(args, model, test_loader)
    print(f'PLCC:{str(plcc)}, SROCC:{str(srocc)}')


if __name__ == '__main__':
    main()
