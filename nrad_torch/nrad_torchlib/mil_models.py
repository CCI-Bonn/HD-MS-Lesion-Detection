"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021
:UPDATED: April 7, 2022 by Chandrakanth Jayachandran Preetha
"""

# adapted from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_resnet import CustomResnet2d
from .custom_resnet import CustomResnet2d_FE
from .custom_resnet import CustomResnet3d


class DeepMILResnet2d(nn.Module):
    def __init__(
        self,
        blocks=[1, 1],
        input_size=(32, 32),
        n_classes=2,
        n_channels=1,
        init_nb_filters=64,
        init_kernel_size=7,
        init_padding=2,
        init_stride=2,
        growth_factor=2,
        batchnorm_eps=1e-5,
        batchnorm_mom=0.1,
        batchnorm_affine=True,
        batchnorm_track_stats=True,
        dilation=1,
        p_dropout=0.5,
        L=512,
        D=128,
        K=1,
    ):
        super(DeepMILResnet2d, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.n_channels = n_channels
        self.input_size = input_size

        self.feature_extractor_part1 = CustomResnet2d_FE(
            blocks=blocks,
            input_size=input_size,
            n_classes=n_classes,
            n_channels=n_channels,
            init_nb_filters=init_nb_filters,
            init_kernel_size=init_kernel_size,
            init_padding=init_padding,
            init_stride=init_stride,
            growth_factor=growth_factor,
            batchnorm_eps=batchnorm_eps,
            batchnorm_mom=batchnorm_mom,
            batchnorm_affine=batchnorm_affine,
            batchnorm_track_stats=batchnorm_track_stats,
            dilation=dilation,
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(
                self.feature_extractor_part1.final_n_filters
                * self.feature_extractor_part1.min_featuremap_dim[0]
                * self.feature_extractor_part1.min_featuremap_dim[1],
                self.L,
            ),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(
            -1,
            self.feature_extractor_part1.final_n_filters
            * self.feature_extractor_part1.min_featuremap_dim[0]
            * self.feature_extractor_part1.min_featuremap_dim[1],
        )
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A


class DeepMILAttention2d(nn.Module):
    def __init__(self, instance_shape=(28, 28), n_channels: int = 3):
        super(DeepMILAttention2d, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.n_channels = n_channels
        self.instance_shape = instance_shape
        self.final_feature_map_size = [
            (((i - 4) // 2) - 2) // 2 for i in instance_shape
        ]

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.n_channels, 36, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(
                48 * self.final_feature_map_size[0] * self.final_feature_map_size[1],
                self.L,
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)

        H = H.view(
            -1, 48 * self.final_feature_map_size[0] * self.final_feature_map_size[1]
        )

        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A


class DeepMILAttention3d(nn.Module):
    def __init__(self, instance_shape=(28, 28, 28), n_channels: int = 3):
        super(DeepMILAttention3d, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.n_channels = n_channels
        self.instance_shape = instance_shape
        self.final_feature_map_size = [
            (((i - 4) // 2) - 2) // 2 for i in instance_shape
        ]

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(self.n_channels, 36, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(
                48
                * self.final_feature_map_size[0]
                * self.final_feature_map_size[1]
                * self.final_feature_map_size[2],
                self.L,
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(
            -1,
            48
            * self.final_feature_map_size[0]
            * self.final_feature_map_size[1]
            * self.final_feature_map_size[1],
        )
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A


class AttentionOnly2d(nn.Module):
    def __init__(self, n_channels: int = 4):
        super(AttentionOnly2d, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.n_channels = n_channels
        self.linear = nn.Sequential(
            nn.Linear(
                self.n_channels,
                self.L,
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        H = x.view(-1, self.n_channels)

        H = self.linear(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

# Classification model with attention layer 
# Implemented by Chandrakanth Jayachandran Preetha, adapted from https://github.com/heykeetae/Self-Attention-GAN
class DeepAttention3d(nn.Module):
    def __init__(self, instance_shape=(160, 160, 160), n_channels: int = 3,n_classes: int = 2):
        super(DeepAttention3d, self).__init__()

        self.n_channels = n_channels
        self.n_classes =  n_classes
        self.instance_shape = instance_shape
        self.attention_feature_map_size = [
            (((((i - 2) // 2) - 2) // 2) -2) // 2 for i in instance_shape]
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(self.n_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        
        self.query = nn.Conv3d(256, 32, kernel_size=1)
        self.key = nn.Conv3d(256, 32, kernel_size=1)
        self.value = nn.Conv3d(256, 256, kernel_size=1)
        

        self.softmax  = nn.Softmax(dim=-1)
        
        self.feature_extractor_part3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        
        self.classifier = nn.Sequential(nn.Linear(512, self.n_classes))

    def forward(self, x):
        
        batchsize, C, width, height, length = x.size()
        
        H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(H)
        H_ = H.view(batchsize, -1,self.attention_feature_map_size[-1]*
                                  self.attention_feature_map_size[-2]*
                                  self.attention_feature_map_size[-3])
        Q = self.query(H)
        K = self.key(H)
        V = self.value(H)
        
        Q_ = Q.view(batchsize, -1, self.attention_feature_map_size[-1]*
                                   self.attention_feature_map_size[-2]*
                                   self.attention_feature_map_size[-3])
        
        K_ = K.view(batchsize, -1, self.attention_feature_map_size[-1]*
                                   self.attention_feature_map_size[-2]*
                                   self.attention_feature_map_size[-3])
        
        V_ = V.view(batchsize, -1, self.attention_feature_map_size[-1]*
                                   self.attention_feature_map_size[-2]*
                                   self.attention_feature_map_size[-3])
                                                  
        A = torch.bmm(Q_.permute(0, 2, 1), K_) 
        A = self.softmax(A)
        
        SA = torch.bmm(V_, A.permute(0, 2, 1)) 
        attn_out = SA+H_
        
        
        attn_out = attn_out.view(batchsize, -1, self.attention_feature_map_size[-1],
                                                self.attention_feature_map_size[-2],
                                                self.attention_feature_map_size[-3])
        
        attn_feature = self.feature_extractor_part3(attn_out)
        attn_feature = torch.flatten(attn_feature, start_dim=1)
        
        Y_prob = self.classifier(attn_feature)
      
        return Y_prob

# Classification model with attention layer, includes batch normalization layers
# Implemented by Chandrakanth Jayachandran Preetha, adapted from https://github.com/heykeetae/Self-Attention-GAN
class DeepAttentionBN3d(nn.Module):
    def __init__(self, instance_shape=(160, 160, 160), n_channels: int = 3,n_classes: int = 2):
        super(DeepAttentionBN3d, self).__init__()

        self.n_channels = n_channels
        self.n_classes =  n_classes
        self.instance_shape = instance_shape
        self.attention_feature_map_size = [
            (((((i - 2) // 2) - 2) // 2) -2) // 2 for i in instance_shape]
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(self.n_channels, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        
        self.query = nn.Conv3d(256, 32, kernel_size=1)
        self.key = nn.Conv3d(256, 32, kernel_size=1)
        self.value = nn.Conv3d(256, 256, kernel_size=1)
        

        self.softmax  = nn.Softmax(dim=-1)
        
        self.feature_extractor_part3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(512, 512, kernel_size=3),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        
        self.classifier = nn.Sequential(nn.Linear(512, self.n_classes))

    def forward(self, x):
        
        batchsize, C, width, height, length = x.size()
        
        H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(H)
        H_ = H.view(batchsize, -1,self.attention_feature_map_size[-1]*
                                  self.attention_feature_map_size[-2]*
                                  self.attention_feature_map_size[-3])
        Q = self.query(H)
        K = self.key(H)
        V = self.value(H)
        
        Q_ = Q.view(batchsize, -1, self.attention_feature_map_size[-1]*
                                   self.attention_feature_map_size[-2]*
                                   self.attention_feature_map_size[-3])
        
        K_ = K.view(batchsize, -1, self.attention_feature_map_size[-1]*
                                   self.attention_feature_map_size[-2]*
                                   self.attention_feature_map_size[-3])
        
        V_ = V.view(batchsize, -1, self.attention_feature_map_size[-1]*
                                   self.attention_feature_map_size[-2]*
                                   self.attention_feature_map_size[-3])
                                                  
        A = torch.bmm(Q_.permute(0, 2, 1), K_) 
        A = self.softmax(A)
        
        SA = torch.bmm(V_, A.permute(0, 2, 1)) 
        attn_out = SA+H_
        
        
        attn_out = attn_out.view(batchsize, -1, self.attention_feature_map_size[-1],
                                                self.attention_feature_map_size[-2],
                                                self.attention_feature_map_size[-3])
        
        attn_feature = self.feature_extractor_part3(attn_out)
        attn_feature = torch.flatten(attn_feature, start_dim=1)
        
        Y_prob = self.classifier(attn_feature)
      
        return Y_prob

