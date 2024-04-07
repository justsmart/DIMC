import torch
import torch.nn as nn
import numpy as np

def test_DIMC(model, mul_X_test, WE_test, args, device):
    model.eval()
    num_X_test = mul_X_test[0].shape[0]
    tmp_q = torch.zeros([num_X_test, args.Nlabel]).to(device)
    index_array_test = np.arange(num_X_test)
    for batch_idx in range(int(np.ceil(num_X_test / args.batch_size))):
        idx = index_array_test[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X_test)]
        mul_X_test_batch = []
        for iv, X in enumerate(mul_X_test):
            mul_X_test_batch.append(X[idx].to(device))

        we = WE_test[idx].to(device)
        _, linshi_q, _, _ = model(mul_X_test_batch, we)
        tmp_q[idx] = linshi_q
        del linshi_q
    # del x0,x1,x2,x3,x4,x5,we  
    # update target distribution p
    yy_pred = tmp_q.data.cpu().numpy()
    yy_pred = np.nan_to_num(yy_pred)
    return yy_pred
