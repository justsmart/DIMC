# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
import h5py
import math
import copy
from loss import Loss
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
from train_sup import train_DIMC
from test_sup import test_DIMC
from measure import *
import time










def filterparam(file_path):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [(float(line.split(' ')[-3]),float(line.split(' ')[-1])) for line in lines]
    return params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--Nlabel', default=7, type=int)
    parser.add_argument('--maxiter', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', type=str, default='corel5k'+'_six_view')
    parser.add_argument('--dataPath', type=str, default='/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k')
    parser.add_argument('--n_z', default=256, type=int)
    # parser.add_argument('--pretrain_path_basis', type=str, default='pascal07/mirflickr')
    parser.add_argument('--MaskRatios', type=float, default=0.5)
    parser.add_argument('--LabelMaskRatio', type=float, default=0.5)
    parser.add_argument('--TraindataRatio', type=float, default=0.7)
    parser.add_argument('--AE_shuffle', type=bool, default=True)
    parser.add_argument('--min_AP', default=0.20, type=float)
    parser.add_argument('--tol', default=1e-7, type=float)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    file_path = 'results-sup/DIMC-sup-' + args.dataset + '_nz_' + str(args.n_z) + '_VMR_' + str(
        args.MaskRatios) + '_LMR_' + str(args.LabelMaskRatio) + '_TR_' + str(
        args.TraindataRatio) + '-best_AP' + '.txt'
    existed_params = filterparam(file_path)

    device = torch.device("cuda" if args.cuda else "cpu")
    Pre_fnum = 10
    pre_momae = [0.9]  # [0.98,0.9,0.95]
    pre_lrkl = [0.1]  # [0.007,0.01,0.0005,0.05,0.1,0.005,0.001]
    pre_alpha = [1e-1]# 1e-1
    best_AUC_me = 0
    best_AUC_mac = 0
    best_AP = 0

    data = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '.mat')
    # data = h5py.File(args.dataPath + '/' + args.dataset + '.mat','r')
    X = data['X'][0]
    # print(X)
    view_num = X.shape[0]
    label = data['label']
    label = np.array(label, 'float32')
    for momae in pre_momae:
        args.momentumkl = momae
        for lrkl in pre_lrkl:
            args.lrkl = lrkl
            for alpha in pre_alpha:
                args.alpha = alpha
                if args.lrkl >= 0.01:
                    args.momentumkl = 0.90

                if (args.lrkl,args.alpha) in existed_params:
                    print('existed param! lr:{} alpha:{}'.format(args.lrkl,args.alpha))
                    # continue
                print(args)
                hm_loss = np.zeros(Pre_fnum)
                one_error = np.zeros(Pre_fnum)
                coverage = np.zeros(Pre_fnum)
                rk_loss = np.zeros(Pre_fnum)
                AP_score = np.zeros(Pre_fnum)

                mac_auc = np.zeros(Pre_fnum)
                auc_me = np.zeros(Pre_fnum)
                mac_f1 = np.zeros(Pre_fnum)
                mic_f1 = np.zeros(Pre_fnum)

                for fnum in range(Pre_fnum):
                    mul_X = [None] * view_num

                    datafold = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '_MaskRatios_' + str(
                        args.MaskRatios) + '_LabelMaskRatio_' +
                                                str(args.LabelMaskRatio) + '_TraindataRatio_' + str(
                        args.TraindataRatio) + '.mat')
                    folds_data = datafold['folds_data']
                    folds_label = datafold['folds_label']
                    folds_sample_index = datafold['folds_sample_index']
                    del datafold
                    Ndata, args.Nlabel = label.shape
                    # training data and test data
                    indexperm = np.array(folds_sample_index[0, fnum], 'int32')
                    train_num = math.ceil(Ndata * args.TraindataRatio)
                    train_index = indexperm[0,0:train_num]-1   #matlab generates the index from '1' to 'Nsample', but python needs from '0' to 'Nsample-1'
                    remain_num = Ndata-train_num
                    val_num = math.ceil(remain_num*0.5)
                    # test_index = indexperm[0, train_num:indexperm.shape[1]] - 1
                    print('val_num',val_num)
                    # print('remain_index',len(test_index))
                    val_index = indexperm[0, train_num:train_num+val_num] - 1
                    
                    rtest_index = indexperm[0, train_num+val_num:indexperm.shape[1]] - 1
                    # test_index = indexperm[0, train_num:indexperm.shape[1]] - 1

                    # incomplete data index    
                    WE = np.array(folds_data[0, fnum], 'int32')
                    # incomplete label construction
                    obrT = np.array(folds_label[0, fnum], 'int32')  # incomplete label index

                    if label.min() == -1:
                        label = (label + 1) * 0.5
                    Inc_label = label * obrT  # incomplete label matrix
                    fan_Inc_label = 1 - Inc_label
                    # incomplete data construction 
                    for iv in range(view_num):
                        mul_X[iv] = np.copy(X[iv])
                        mul_X[iv] = mul_X[iv].astype(np.float32)
                        WEiv = WE[:, iv]
                        ind_1 = np.where(WEiv == 1)
                        ind_1 = (np.array(ind_1)).reshape(-1)
                        ind_0 = np.where(WEiv == 0)
                        ind_0 = (np.array(ind_0)).reshape(-1)
                        mul_X[iv][ind_1, :] = StandardScaler().fit_transform(mul_X[iv][ind_1, :])
                        mul_X[iv][ind_0, :] = 0
                        clum = abs(mul_X[iv]).sum(0)
                        ind_11 = np.array(np.where(clum != 0)).reshape(-1)
                        new_X = np.copy(mul_X[iv][:, ind_11])
                        # del X0
                        mul_X[iv] = torch.Tensor(np.nan_to_num(np.copy(new_X)))
                        del new_X, ind_0, ind_1, ind_11, clum

                    WE = torch.Tensor(WE)
                    mul_X_val = [xiv[val_index] for xiv in mul_X]
                    mul_X_rtest = [xiv[rtest_index] for xiv in mul_X]
                    mul_X_train = [xiv[train_index] for xiv in mul_X]
                    WE_val = WE[val_index]
                    WE_rtest = WE[rtest_index]
                    WE_train = WE[train_index]
                    obrT = torch.Tensor(obrT)
                    
                    # Inc_label = torch.Tensor(Inc_label)
                    # fan_Inc_label = torch.Tensor(fan_Inc_label)
                    # args.n_input = [X0.shape[1],X1.shape[1],X2.shape[1],X3.shape[1],X4.shape[1],X5.shape[1]]
                    args.n_input = [xiv.shape[1] for xiv in mul_X]

                    yv_label = np.copy(label[val_index])
                    yrt_label = np.copy(label[rtest_index])
                    train_label = torch.Tensor(label[train_index])
                    train_obrT = torch.Tensor(obrT[train_index])
                    
                    ind_00_val = np.array(np.where(abs(yv_label).sum(1) == 0)).reshape(-1)
                    ind_00_test = np.array(np.where(abs(yrt_label).sum(1) == 0)).reshape(-1)
                    
                    model, value_result,all_results = train_DIMC(mul_X_train, mul_X_val,WE_train,WE_val,train_label,yv_label,ind_00_val,train_obrT, device,args)
                    # np.save(args.dataset+'_V_'+str(args.MaskRatios)+'_L_'+str(args.LabelMaskRatio)+'_'+str(fnum)+'.npy', np.array(all_results))
                    yp_prob = test_DIMC(model,mul_X_test,WE_test,args,device)
                    yp_prob = np.delete(yp_prob,ind_00_test,axis=0)
                    value_result = do_metric(yp_prob,yt_label)
                    print(
                        "final:hamming-loss" + ' ' + "one-error" + ' ' + "coverage" + ' ' + "ranking-loss" + ' ' + "average-precision" + ' ' + "macro-auc" + ' ' + "auc_me" + ' ' + "macro_f1" + ' ' + "micro_f1")
                    print(value_result)

                    hm_loss[fnum] = value_result[0]
                    one_error[fnum] = value_result[1]
                    coverage[fnum] = value_result[2]
                    rk_loss[fnum] = value_result[3]
                    AP_score[fnum] = value_result[4]
                    mac_auc[fnum] = value_result[5]
                    auc_me[fnum] = value_result[6]
                    mac_f1[fnum] = value_result[7]
                    mic_f1[fnum] = value_result[8]
                if AP_score.mean() > best_AP:
                    best_AP = AP_score.mean()

                file_handle = open(file_path, mode='a')
                if os.path.getsize(file_path) == 0:
                    file_handle.write(
                        'mean_AP std_AP mean_hamming_loss std_hamming_loss mean_ranking_loss std_ranking_loss mean_AUCme std_AUCme  mean_one_error std_one_error mean_coverage std_coverage mean_macAUC std_macAUC mean_macro_f1 std_macro_f1 mean_micro_f1 std_micro_f1 lrkl momentumKL alphakl\n')

                file_handle.write(str(round(AP_score.mean(),4)) + ' ' +
                                    str(round(AP_score.std(),4)) + ' ' +
                                    str(round(hm_loss.mean(),4)) + ' ' +
                                    str(round(hm_loss.std(),4)) + ' ' +
                                    str(round(rk_loss.mean(),4)) + ' ' +
                                    str(round(rk_loss.std(),4)) + ' ' +
                                    str(round(auc_me.mean(),4)) + ' ' +
                                    str(round(auc_me.std(),4)) + ' ' +
                                    str(round(one_error.mean(),4)) + ' ' +
                                    str(round(one_error.std(),4)) + ' ' +
                                    str(round(coverage.mean(),4)) + ' ' +
                                    str(round(coverage.std(),4)) + ' ' +
                                    str(round(mac_auc.mean(),4)) + ' ' +
                                    str(round(mac_auc.std(),4)) + ' ' +
                                    str(round(mac_f1.mean(),4)) + ' ' +
                                    str(round(mac_f1.std(),4)) + ' ' +
                                    str(round(mic_f1.mean(),4)) + ' ' +
                                    str(round(mic_f1.std(),4)) + ' ' +
                                    str(args.lrkl) + ' ' +
                                    str(args.momentumkl) + ' ' +
                                    str(args.alpha)


                                    )

                file_handle.write('\n')
                file_handle.close()
