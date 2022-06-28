# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, ini=0.5):
        """
        Args:
            num:需要自适应权重的损失数量
            ini:初始化，为超参数，可调
        """
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True).cuda()
        params[0] = params[0] * ini
        params[1] = params[1] * ini
        self.params = torch.nn.Parameter(params)
        # self.params = self.params.cuda()

    def forward(self, *x):
        # print('*x = ', *x)
        loss_sum = 0
        for i, loss in enumerate(x):
            # print('i = {}, loss = {}'.format(i, loss))
            # print('self.params_{} = {}'.format(i, self.params[i]))
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            # print('loss_sum = ', loss_sum)
        return loss_sum


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    # AutomaticWeightedLoss.forward(1)

    print(awl.parameters())
