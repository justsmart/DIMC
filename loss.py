import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, device):
        super(Loss, self).__init__()


        


    def wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret
    def weighted_CL_loss(self,sub_target,target_pre,sub_obrT):
        return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                            + (1-sub_target).mul(torch.log(1 - target_pre + 1e-10))).mul(sub_obrT)))



