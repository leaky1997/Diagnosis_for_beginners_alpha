# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:38:36 2020

@author: 李奇
"""
from torch.autograd import Function


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx:ctx是一个上下文对象，可用于存储信息以进行反向计算
        """
        output = grad_output.neg() * ctx.alpha

        return output, None


def grad_reverse(x, lambd=1.0):
    return ReverseLayer.apply(x, lambd)
