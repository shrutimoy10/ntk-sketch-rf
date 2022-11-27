import math
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
import torch.nn as nn
from scipy.special import erf
from learn_ntk_utils import linear_chi_square
# from ntk_random_features import AcosFeatureMap, NtkFeatureMapOps




# class AcosFeatureMap(nn.Module):

#     def __init__(self, input_dim, output_dim, W,dev='cpu'):
#         super(AcosFeatureMap, self).__init__()
#         if input_dim == 0 or output_dim == 0:
#             import pdb
#             pdb.set_trace()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         # self.do_leverage = do_leverage
#         # if not do_leverage:
#         #     self.W = torch.randn(input_dim, output_dim)
#         #     self.norm_const = math.sqrt(2.0 / self.output_dim)
#         # else:
#         #     # self.W = gibbs_sampling_weighted_normal(input_dim, output_dim)
#         #     # W_norm = self.W.pow(2).sum(0).sqrt()
#         #     # self.norm_const = math.sqrt(2.0 * input_dim / output_dim) / W_norm
#         #     # self.norm_const = self.norm_const.to(dev)
#         #     '''
#         #     Without using the leverage score sampling, we want
#         #     to check whether the kernel optimization will learn the
#         #     required distribution.
#         #     '''
#         #     self.W = torch.randn(input_dim, output_dim)
#         #     self.norm_const = math.sqrt(2.0 / self.output_dim)
#         '''
#             Without using the leverage score sampling, we want
#             to check whether the kernel optimization will learn the
#             required distribution.
#         '''
#         # self.W = torch.randn(input_dim, output_dim)
#         self.W = W
#         self.norm_const = math.sqrt(2.0 / self.output_dim)

#         self.W = self.W.to(dev)

#     def forward(self, x, order):
#         try:
#             assert x.shape[1] == self.input_dim
#         except:
#             import pdb
#             pdb.set_trace()
#         # if self.do_leverage:
#         #     assert order == 1
#         xw = x @ self.W
#         if order == 0:
#             return (xw > 0) * self.norm_const
#         elif order == 1:
#             return (abs(xw) + xw) * (self.norm_const / 2.0)
#         else:
#             raise NotImplementedError



def optimizearccosfeatures(x, y, W, Nw, rho, tol, input_dim, output_dim, norm_const, order):
    '''
    Optimizes the random features generated for the arccos0 and
    arccos1 kernels. We use these optimized features for computing
    the NTK
    Here,
    m1, m0 are the output dimensions for the acos1 and acos0 kernels.
    ms is the output dimension of the POLYSKETCH technique.
    x is the output of the previous layer, such as the k_nngp or k_ntk
    '''
    n, d = x.shape
    '''
    First we optimize for the arccos0 kernel.
    Compute for Nw features, then according to the distribution
    alpha, trim those features to m0.
    '''
    # arccos_0 = AcosFeatureMap(d, Nw, W_0, 'cuda:0')
    xw = x @ W
    if order == 0:
        phi = (xw > 0) * norm_const
    elif order == 1:
        phi = (abs(xw) + xw) * (norm_const / 2.0)

    # phi_0 = arccos_0(x, order = order)
    # print("Phi  shape : {}, y shape : {}".format(phi.shape, y.shape))
    # Ks = torch.dot(torch.transpose(phi, 0, 1), y)
    # print("phi.T : ", torch.transpose(phi, 0, 1).float())
    Ks = torch.transpose(phi, 0, 1).float() @ y.float()
    Ks = torch.square(Ks)

    Ks = Ks.reshape(Ks.shape[0])
    Ks = Ks.cpu().detach().numpy()

    # print("Ks shape : {}".format(Ks.shape))

    u = np.ones(Nw) * (1/Nw)

    alpha_temp = linear_chi_square(-Ks, u, rho/Nw, tol)


    #taking the non zero probabilities
    # print("Sum Alpha_temp: {}, Length alpha_temp : {}".format(sum(alpha_temp), len(alpha_temp)))
    idx = np.argwhere(alpha_temp > 0)
    opt_rf_num = len(idx)  # optimal number of random features as returned by the optimization
    # print("Length IDX : {}".format(len(idx)))
    alpha = alpha_temp[idx]
    W_opt = W[:, idx]
    W_opt = W_opt.reshape((W_opt.shape[0], W_opt.shape[1]))
    print(" In arccos py W_opt shape : {}, opt_rf_num : {}".format(W_opt.shape, opt_rf_num))
    if order == 1:
        norm_const_opt = norm_const[idx]
    else:
        norm_const_opt = norm_const

    # print("Shape of W_opt : ", W_opt.shape)
    # if len(W_opt) > output_dim:
    #     W_opt = W_opt[:, :output_dim]



    alpha_dist = np.cumsum(alpha/sum(alpha))

    return W_opt, norm_const_opt, opt_rf_num,alpha, alpha_dist

    