"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
""" 
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.nn import NLLLoss
import torch


class MILNegLogBernoulli(object):
    def __call__(self, model_out, Y):
        Y = Y.float()
        Y_prob, _, _ = model_out
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )  # negative log bernoulli
        return neg_log_likelihood

    
class BCE_loss(BCELoss):
    def __init__(self, weight = None) -> None:
        super(BCELoss, self).__init__()

    def __call__(self, model_out, Y):
        Y = Y.float()
        Y_prob, _, _  = model_out
        return BCELoss()(Y_prob,Y)
