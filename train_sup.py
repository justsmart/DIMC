from torch.optim import Adam, SGD
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DIMCNet
from loss import Loss
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np
from test_sup import test_DIMC
import copy
from measure import *
import time

def train_DIMC(mul_X, mul_X_test, WE,WE_test,label,yt_label,ind_00,obrT,device,args):
    # return None, torch.randn(9, 1)
    # print(mul_X[0].shape,mul_X_test[0].shape)
    yt_label = np.delete(yt_label, ind_00, axis=0)
    model = DIMCNet(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.n_z,
        Nlabel=args.Nlabel).to(device)
    loss_model = Loss( device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():

                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)
    num_X = mul_X[0].shape[0]
    num_X_test = mul_X_test[0].shape[0]
    print(num_X, num_X_test)
    optimizer = SGD(model.parameters(), lr=args.lrkl, momentum=args.momentumkl)
    # optimizer = Adam(model.parameters(), lr=args.lrkl)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    total_loss = 0
    ytest_Lab = np.zeros([mul_X_test[0].shape[0], args.Nlabel])
    ap_loss = []
    best_value_result = [0] * 10
    best_value_epoch = 0
    best_train_model = copy.deepcopy(model)
    
    for epoch in range(int(args.maxiter)):
        model.train()
        total_loss_last = total_loss
        total_loss = 0
        ytest_Lab_last = np.copy(ytest_Lab)
        index_array = np.arange(num_X)
        if args.AE_shuffle == True:
            np.random.shuffle(index_array)
        tt=time.time()
        for batch_idx in range(int(np.ceil(num_X / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X)]
            mul_X_batch = []
            for iv, X in enumerate(mul_X):
                mul_X_batch.append(X[idx].to(device))

            we = WE[idx].to(device)
            sub_target = (label[idx]*obrT[idx]).to(device)
            # fan_sub_target = fan_Inc_label[idx].to(device)
            sub_obrT = obrT[idx].to(device)
            optimizer.zero_grad()

            x_bar_list, target_pre, fusion_z, individual_zs = model(mul_X_batch, we)


            loss_CL = loss_model.weighted_CL_loss(sub_target,target_pre,sub_obrT)
            loss_AE = 0
            for iv, x_bar in enumerate(x_bar_list):
                loss_AE += loss_model.wmse_loss(x_bar, mul_X_batch[iv], we[:, iv])
            fusion_loss = loss_CL + args.alpha * loss_AE
            # print('all:',fusion_loss.item())
            total_loss += fusion_loss.item()
            fusion_loss.backward()
            optimizer.step()
        # scheduler.step()
        # print('traintime:',time.time()-tt)
        st = time.time()
        yp_prob = test_DIMC(model, mul_X_test, WE_test, args, device)
        # print('testtime:',time.time()-st)
        yp_prob = np.delete(yp_prob, ind_00, axis=0)
        value_result = do_metric(yp_prob, yt_label)
        ap_loss.append([value_result[4],total_loss])
        total_loss = total_loss / (batch_idx + 1)
        print("sup_epoch {} loss={:.4f} hamming loss={:.4f} AP={:.4f} AUC={:.4f} auc_me={:.4f}"
              .format(epoch, total_loss, value_result[0], value_result[4], value_result[5], value_result[6]))
        if best_value_result[4]+best_value_result[3]*0.5 < value_result[4]+value_result[3]*0.5:
            best_value_result = value_result
            best_train_model = copy.deepcopy(model)
            best_value_epoch = epoch

            # torch.save(model)
        ytest_Lab = yp_prob > 0.5
        del yp_prob
        # delta_y = np.sum(ytest_Lab != ytest_Lab_last).astype(np.float32) / ytest_Lab.shape[0] / ytest_Lab.shape[1]
        if epoch > 100 and ( (best_value_result[4]-value_result[4]>0.03) or
                best_value_result[4] < args.min_AP or (abs(total_loss_last - total_loss) < 1e-7)):
            print('Training stopped: epoch=%d, best_epoch=%d, best_AP=%.7f, min_AP=%.7f,total_loss=%.7f' % (
                epoch, best_value_epoch, best_value_result[4], args.min_AP, total_loss))
            break

    return best_train_model, best_value_result,ap_loss