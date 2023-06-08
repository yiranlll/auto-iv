import torch
import model
from train import train, test
from csv_dataset import CustomCSVDataset
from utils import Log
import pandas as pd
import glob
import random

from torch.utils.data import DataLoader
from datetime import datetime


if __name__ == '__main__':
    device = torch.device("cuda")

    v_dim, x_dim, y_dim = 6, 2, 1
    epochs = 20
    lr = 1e-3

    parameter = {
        'z_dim': 2,
        'c_dim': 2,
        'emb_dim': 2,
        'lld_weight': [1, 1, 1, 1, 1],
        'mi_weight': [1, 1, 1, 1, 1],
        'n_layer':2,
        'alpha': 0.5,
        'eta': 0.5
    }

    auto_iv = model.AutoIV(x_dim, v_dim, y_dim, parameter).to(device)
    log = Log(log_path="./log/")

    files = glob.glob("data/*.csv")
    random.seed(42)
    random.shuffle(files)
    train_dataset = CustomCSVDataset(files[:len(files)//2], 1)
    test_dataset = CustomCSVDataset(files[len(files)//2:], 1)
    
    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)
    optimizer_list = {
        'c_rep': torch.optim.Adam(list(auto_iv.c_rep.parameters()), lr=lr),
        'z_rep': torch.optim.Adam(list(auto_iv.z_rep.parameters()), lr=lr),
        'reg_x': torch.optim.Adam(list(auto_iv.regressionX.parameters()), lr=lr),
        'reg_y': torch.optim.Adam(list(auto_iv.regressionY.parameters()), lr=lr),
        'minet': torch.optim.Adam(list(auto_iv.cx_minet.parameters()) + list(auto_iv.cy_minet.parameters()) + list(auto_iv.zc_minet.parameters())+list(auto_iv.zx_minet.parameters())+list(auto_iv.zy_minet.parameters()), lr=lr)
    }
    log.write("Experiment Parameters\n")
    for k, v in parameter.items():
        log.write(k + ": {}".format(v))
    for t in range(epochs):
        log.write("\nEpoch {}-------------------------------".format(t+1))
        train(train_dataloader, auto_iv, optimizer_list, log)
        test(test_dataloader, auto_iv, log)
    print("Done!")

    path = "./saved_models/"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    torch.save(auto_iv.state_dict(), path+"auto_iv_"+now+".pth")