#-*-coding:gb2312-*-
"""
  This script provides an exmaple to the fine-tuning and self-distillation 
  peocess of the FastBERT.
"""
import os, sys
import torch
import time
import json
import random
import argparse
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model
from uer.layers.multi_headed_attn import MultiHeadedAttention
import numpy as np
import time
from thop import profile

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('./log')


torch.set_num_threads(1)

np.set_printoptions(suppress=True)


def normal_shannon_entropy(p, labels_num):
    entropy = torch.distributions.Categorical(probs=p).entropy()
    normal = -np.log(1.0 / labels_num)
    return entropy / normal

class ClassifierFC(nn.Module):

    def __init__(self, args, input_size, labels_num):
        super(ClassifierFC, self).__init__()
        self.input_size = input_size
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.labels_num = labels_num
        self.pooling = args.pooling
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, args.dropout)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)
        self.output_layer_3 = nn.Linear(self.input_size, self.input_size)
        self.output_layer_4 = nn.Linear(self.input_size, labels_num)
        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        self.fc = nn.Linear(768, args.labels_num)
        self.dropout = nn.Dropout(0.5)

        # Capsule
        # self.input_dim_capsule=args.hidden_size
        self.input_dim_capsule = self.input_size
        # self.input_dim_capsule=256
        self.num_capsule = 5
        self.dim_capsule = 10
        self.routings = 3
        self.kernel_size = (9, 1)  #
        self.share_weights = True
        self.activation = 'default'
        if self.activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(BATCH_SIZE, self.input_dim_capsule, self.num_capsule * self.dim_capsule))
        self.dropout1 = nn.Dropout(0.25)
        self.fc_capsule = nn.Linear(self.num_capsule * self.dim_capsule,
                                    self.labels_num)  # num_capsule*dim_capsule -> num_classes

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # print(x.size())
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x = F.max_pool1d(x, x.size(2))
        return x

    def Capsule(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
            # print(u_hat_vecs.size())
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        # print(u_hat_vecs.size())
        # print(u_hat_vecs.size())
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            # print(b.szie())
            b = b.permute(0, 2, 1)
            #print(b.szie())
            c = F.softmax(b, dim=2)
            #print(c.szie())
            c = c.permute(0, 2, 1)
            #print(c.szie())
            b = b.permute(0, 2, 1)
            #print(b.szie())
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # print(outputs.szie())
            # print(outputs.size())
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
                # print(b.szie())
                # print(b.size())
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale

    def forward(self, hidden, mask):
        #print('do it sssss')
        # hidden = torch.tanh(self.output_layer_0(hidden))
        # hidden = self.self_atten(hidden, hidden, hidden, mask)
        
        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]
        '''
        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        '''
        output_1 = torch.tanh(self.output_layer_3(hidden))
        # logits = self.output_layer_2(output_1)
        logits = self.output_layer_4(output_1)
        '''
        #CNN
        out = hidden.unsqueeze(1)
        print(out.size())
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out=out.view(out.shape[0],self.input_dim_capsule)
        print(out.size())
        output = self.dropout(out)
        logits = self.fc(out)
        #Capsule
        output_capsule=self.Capsule(hidden)
        #print(output_capsule.shape)
        output_capsule=output_capsule.view(output_capsule.shape[0],-1)
        output_capsule=self.dropout1(output_capsule)
        logits=self.fc_capsule(output_capsule)
        '''
        return logits, None

class ClassifierCap(nn.Module):

    def __init__(self, args, input_size, labels_num):
        super(ClassifierCap, self).__init__()
        self.input_size = input_size
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.labels_num = labels_num
        self.pooling = args.pooling
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, args.dropout)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)
        self.output_layer_3 = nn.Linear(self.input_size, self.input_size)
        self.output_layer_4 = nn.Linear(self.input_size, labels_num)
        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        self.fc = nn.Linear(768, args.labels_num)
        self.dropout = nn.Dropout(0.5)

        # Capsule
        # self.input_dim_capsule=args.hidden_size
        self.input_dim_capsule = self.input_size
        # self.input_dim_capsule=256
        self.num_capsule = 5
        self.dim_capsule = 10
        self.routings = 3
        self.kernel_size = (9, 1)  #
        self.share_weights = True
        self.activation = 'default'
        if self.activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(BATCH_SIZE, self.input_dim_capsule, self.num_capsule * self.dim_capsule))
        self.dropout1 = nn.Dropout(0.25)
        self.fc_capsule = nn.Linear(self.num_capsule * self.dim_capsule,
                                    self.labels_num)  # num_capsule*dim_capsule -> num_classes

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # print(x.size())
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x = F.max_pool1d(x, x.size(2))
        return x

    def Capsule(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
            # print(u_hat_vecs.size())
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        # print(u_hat_vecs.size())
        # print(u_hat_vecs.size())
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            # print(b.szie())
            b = b.permute(0, 2, 1)
            #print(b.szie())
            c = F.softmax(b, dim=2)
            #print(c.szie())
            c = c.permute(0, 2, 1)
            #print(c.szie())
            b = b.permute(0, 2, 1)
            #print(b.szie())
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # print(outputs.szie())
            # print(outputs.size())
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
                # print(b.szie())
                # print(b.size())
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale

    def forward(self, hidden, mask):
        #print('do it sssss')
        # hidden = torch.tanh(self.output_layer_0(hidden))
        # hidden = self.self_atten(hidden, hidden, hidden, mask)
        '''
        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]
        
        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        
        output_1 = torch.tanh(self.output_layer_3(hidden))
        # logits = self.output_layer_2(output_1)
        logits = self.output_layer_4(output_1)
        #
        #CNN
        out = hidden.unsqueeze(1)
        print(out.size())
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out=out.view(out.shape[0],self.input_dim_capsule)
        print(out.size())
        output = self.dropout(out)
        logits = self.fc(out)
        '''
        
        #Capsule
        output_capsule=self.Capsule(hidden)
        #print(output_capsule.shape)
        output_capsule=output_capsule.view(output_capsule.shape[0],-1)
        output_capsule=self.dropout1(output_capsule)
        logits=self.fc_capsule(output_capsule)
        
        return logits, None


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    # print(ip[0])
    # probility= nn.functional.softmax(ip, dim=1)
    # print(probility[0])
    # entropys = normal_shannon_entropy(probility, 31)
    # en = torch.mean(entropys, dim=-1)
    # print(entropys[0])
    #
    w1 = torch.norm(x1, 2, dim)
    # print(w1[0])
    w2 = torch.norm(x2, 2, dim)
    # print(w2[0])
    return ip / torch.ger(w1, w2).clamp(min=eps), ip


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.15):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, 50))
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        self.fc = nn.Linear(768, self.out_features)
        self.dropout = nn.Dropout(0.5)
        self.output_layer_pooling = nn.Linear(self.in_features, self.in_features)
        # Capsule
        # self.input_dim_capsule=args.hidden_size
        self.input_dim_capsule = self.in_features
        # self.input_dim_capsule=256
        self.num_capsule = 5
        self.dim_capsule = 10
        self.routings = 3
        self.kernel_size = (9, 1)  #
        self.share_weights = True
        self.activation = 'default'
        if self.activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(BATCH_SIZE, self.input_dim_capsule, self.num_capsule * self.dim_capsule))
        self.dropout1 = nn.Dropout(0.25)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # print(x.size())
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x = F.max_pool1d(x, x.size(2))
        return x

    def Capsule(self, x):
        '''
        print('SHOW some param of capsule')
        print('input x: {}'.format(x.shape))  # x [16,128,768]
        print('W: {}'.format(self.W.shape))   # W [1,768,50]
        print('input_dim_capsule: {}'.format(self.input_dim_capsule)) #768
        '''
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)  #u [16,128,50]
            #print('after x * w -> u: {} '.format(u_hat_vecs.shape))
        else:
            print('add later')

        batch_size = x.size(0)
        #print('bs : {}'.format(batch_size))
        input_num_capsule = x.size(1) # 128
        #print('input_num_capsule : {}'.format(input_num_capsule))
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))  #[16,128,5,10]
        #print('after reshape u: {}'.format(u_hat_vecs.shape))
        # print(u_hat_vecs.size())
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # (batch_size,num_capsule,input_num_capsule,dim_capsule)[16,5,128,10]
        #print('after permute u:{}'.format(u_hat_vecs.shape))
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule) [16,5,128]
        #print('pro b :{}'.format(b.shape))
        for i in range(self.routings):
            # print(b.szie())
            b = b.permute(0, 2, 1)  #每个dim对各个胶囊的softmax
            # print(b.szie())
            c = F.softmax(b, dim=2)
            # print(c.szie())
            c = c.permute(0, 2, 1)  
            #print('after softmax and permute c: {}'.format(c.shape))
            b = b.permute(0, 2, 1)
            # print(b.szie())
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            #print('outputs shape:{}'.format(outputs.shape))
            # print(outputs.szie())
            # print(outputs.size())
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
                # print(b.szie())
                # print(b.size())
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale

    def forward(self, hidden, label):
        '''
        out = hidden.unsqueeze(1)
        # print(out.size())
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = out.view(out.shape[0], self.in_features)
        # print(out.size())
        input = self.dropout(out)
        '''
        # Capsule
        output_capsule = self.Capsule(hidden)
        #print(output_capsule.shape)
        output_capsule = output_capsule.view(output_capsule.shape[0], -1)
        output_capsule = self.dropout1(output_capsule)
        input = output_capsule
        
        # logits = self.fc_capsule(output_capsule)
        #input = hidden[:, 0, :]
        #input = torch.tanh(self.output_layer_pooling(input))#[CLS] pooling
        cosine, ip = cosine_sim(input, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output, cosine, ip

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class FastBertClassifier(nn.Module):
    def __init__(self, args, model):
        super(FastBertClassifier, self).__init__()
        
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.classifiers = nn.ModuleList([
            MarginCosineProduct(args.hidden_size, self.labels_num) \
            for i in range(self.encoder.layers_num)
        ])#capsule+lmcl
        self.classifiers12 =ClassifierFC(args, args.hidden_size, self.labels_num) #CE
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.soft_criterion = nn.KLDivLoss(reduction='batchmean')
        self.threshold = args.speed
        self.loss_mse = nn.MSELoss()
        # self.encoder.layers_num=12

    def Focal_Loss(self, class_num, inputs, targets, alpha=None, gamma=2, size_average=False):
        # alpha = Variable(torch.ones(class_num, 1))
        # num = [1914,1896,1881,1876,1852,1847,1818]
        num = [71, 32, 32, 32, 40, 358, 24, 32, 143, 83, 74, 32, 91, 32, 84, 88, 78, 32, 136, 32, 45, 38, 95, 84, 94,
               82, 94, 242, 88, 72, 609]
        total_num = float(sum(num))
        classes_w_t1 = [total_num / ff for ff in num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]
        alpha = torch.tensor(classes_w_t2)
        if isinstance(alpha, Variable):
            alpha = alpha
        else:
            alpha = Variable(alpha)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if inputs.is_cuda and not alpha.is_cuda:
            alpha = alpha.cuda()
        alpha = alpha[ids.data]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        if size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    def In(self, a, b):
        c = a == b
        c = c.numpy().astype(int)
        c = torch.from_numpy(c)
        # c=c.from
        return c

    def forward(self, src, label, mask, fast=True, teacher=True):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)

        # Encoder.
        seq_length = emb.size(1)
        mask = (mask > 0). \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        if self.training:

            if teacher:

                # training main part of the model
                hidden = emb
                batch_size = emb.shape[0]
                logits_list = []
                hidden_list=[]
                loss = 0
                teacher_loss=0
                for i in range(self.encoder.layers_num):
                    hidden = self.encoder.transformer[i](hidden, mask)
                    hidden_list.append(hidden)
                    
                    #if i==self.encoder.layers_num-1:
                    #    logits, _= self.classifiers12(hidden, mask)

                    #else:
                        #hidden = self.encoder.transformer[i](hidden, mask)
                        #logits,_ = self.classifiers[i](hidden, mask)#FC
                    # logits_list.append(logits)
                teacher_logits, _, _ = self.classifiers[i](hidden, label)
                teacher_loss += self.criterion(self.softmax(teacher_logits.view(-1, self.labels_num)), label.view(-1))  # LMCL
                return teacher_loss, teacher_logits
            else:

                # distillate the subclassifiers
                loss, hidden, hidden_list = 0, emb, []

                for i in range(self.encoder.layers_num):
                    hidden = self.encoder.transformer[i](hidden, mask)
                    hidden_list.append(hidden)
                # teacher_logits = self.classifiers[-1](hidden_list[-1], mask).view(-1, self.labels_num)
                teacher_logits, teacher_feature = self.classifiers[-1](hidden_list[-1], mask)
                teacher_probs = nn.functional.softmax(teacher_logits, dim=1)
                loss = 0
                for i in range(self.encoder.layers_num - 1):
                    # student_logits,_= self.classifiers[i](hidden, mask)
                    student_logits, student_feature = self.classifiers[i](hidden_list[i], mask)
                    loss += self.soft_criterion(self.softmax(student_logits), teacher_probs)
                    loss += self.criterion(self.softmax(student_logits.view(-1, self.labels_num)), label.view(-1))
                    # loss+=0.1*self.loss_mse(student_feature,teacher_feature)
                loss += self.criterion(self.softmax(teacher_logits.view(-1, self.labels_num)), label.view(-1))
                return loss, teacher_logits

        else:
            # inference
            if fast:
                # fast mode
                hidden = emb  # (batch_size, seq_len, emb_size)
                batch_size = hidden.size(0)
                logits = torch.zeros(batch_size, self.labels_num, dtype=hidden.dtype, device=hidden.device)
                num_layer = np.zeros(12, dtype=np.int32)
                en_means = []
                abs_diff_idxs = torch.arange(0, batch_size, dtype=torch.long, device=hidden.device)
                for i in range(self.encoder.layers_num):
                    # for i in range(1):
                    # print(i)
                    num = len(abs_diff_idxs)
                    hidden = self.encoder.transformer[i](hidden, mask)

                    logits_loss, logits_this_layer, ip = self.classifiers[i](hidden, label)  # (batch_size, labels_num)
                    # print(i)
                    # print(nn.Softmax(dim=1)( logits_this_layer))
                    logits[abs_diff_idxs] = logits_this_layer

                    # filter easy sample
                    abs_diff_idxs, rel_diff_idxs, en_mean = self._difficult_samples_idxs(abs_diff_idxs,
                                                                                         logits_this_layer, i, ip)
                    hidden = hidden[rel_diff_idxs, :, :]
                    en_means.append(en_mean.cpu())
                    mask = mask[rel_diff_idxs, :, :]
                    label = label[rel_diff_idxs]

                    num_layer[i] = num - len(abs_diff_idxs)

                    if len(abs_diff_idxs) == 0:
                        break

                # print(num_layer)
                return None, logits, num_layer, en_means
            else:
                # normal mode
                hidden = emb
                logits_list = []
                self.encoder.layers_num = 12
                for i in range(self.encoder.layers_num):
                    hidden = self.encoder.transformer[i](hidden, mask)
                    #if i==11:
                    #    logits, _= self.classifiers12(hidden, mask)
                    #    logits_list.append(logits)
                    #else:
                        #hidden = self.encoder.transformer[i](hidden, mask)
                    #logits,_ = self.classifiers[i](hidden, mask)#FC
                    #logits_list.append(logits)
                        #logits, _, _ = self.classifiers[i](hidden, label)
                logits, cosine, _ = self.classifiers[i](hidden, label)
                logits_list.append(cosine)
                #logits,  _ = self.classifiers12(hidden, label)
                # probs = nn.Softmax(dim=1)(cosine)
                # entropys = normal_shannon_entropy(probs, self.labels_num)
                # en = torch.mean(entropys, dim=-1)
                # print(en)
                
                #logits_list.append(logits)
                return None, logits_list

    def _difficult_samples_idxs(self, idxs, logits, i, ip):
        # logits: (batch_size, labels_num)
        probs = nn.Softmax(dim=1)(ip)
        entropys = normal_shannon_entropy(probs, self.labels_num)
        en = torch.mean(entropys, dim=-1)
        #if i==0:
        #   print(entropys)
        ## torch.nonzero() is very time-consuming on GPU
        # Please see https://github.com/pytorch/pytorch/issues/14848
        # If anyone can optimize this operation, please contact me, thank you!
        # rel_diff_idxs = (entropys > self.threshold).nonzero().view(-1)
        rel_diff_idxs = (entropys > self.threshold).nonzero().view(-1)
        abs_diff_idxs = torch.tensor([idxs[i] for i in rel_diff_idxs], device=logits.device)
        return abs_diff_idxs, rel_diff_idxs, en


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="./models/roberta_to_uer_model.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_teacher_model_path", default="./models/fastbert.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--output_student_model_path", default="./models/fastbert.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_zh_vocab.txt",type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")

    #parser.add_argument("--dev_path", type=str, required=True, help="Path of the devset.")

    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", default="bert",choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                         help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=15,
                        help="Number of epochs.")
    parser.add_argument("--distill_epochs_num", type=int, default=10,
                        help="Number of distillation epochs.")
    parser.add_argument("--report_steps", type=int, default=70,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")
    parser.add_argument("--fast_mode", dest='fast_mode', action='store_true', help="Whether turn on fast mode")
    parser.add_argument("--speed", type=float, default=0.9523,
                        help="Threshold of Uncertainty, i.e., the Speed in paper.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set)
    print(labels_set)
    print(args.labels_num)
    '''
    labels_dict={}
    train_dict={}
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                #label = int(line[columns["label"]])-1
                #labels_set.add(label)
                label = line[columns["label"]]
                #print(label)
                labels_dict[label]=labels_dict.get(label,len(labels_dict))
                la=labels_dict.get(label)-1
                train_dict[la]=train_dict.get(la,0)+1
            except:
                pass
    args.labels_num = len(labels_dict)-1
    print(labels_dict)
    print(train_dict)
    print(args.labels_num)
    '''
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification model.
    start = time.time()
    model = FastBertClassifier(args, model)
    end = time.time()
    print('loadmodel:' + str(end - start))
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    '''
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Read dataset.
    def read_dataset(path, set):
        dataset = []
        # a=[1,5,10,15,20]
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                try:
                    line = line.strip().split('\t')
                    if len(line) == 2:
                        label = int(line[columns["label"]])

                        '''
                        label =line[columns["label"]]
                        label=labels_dict.get(label)-1
                        #print(label)
                        if label==-1:
                            #print(label)
                            continue
                        '''
                        text = line[columns["text_a"]]
                        tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
                        tokens = [CLS_ID] + tokens
                        mask = [1] * len(tokens)
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 3:  # For sentence pair input.
                        label = int(line[columns["label"]])
                        text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 4:  # For dbqa input.
                        qid = int(line[columns["qid"]])
                        label = int(line[columns["label"]])
                        text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask, qid))
                    else:
                        pass

                except:
                    pass
        return dataset

    # Evaluation function.
    def evaluate(args, is_test, fast_mode=False):
        if is_test:
            dataset = read_dataset(args.test_path, True)
        else:
            dataset = read_dataset(args.dev_path, False)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]

        print("The number of evaluation instances: ", instances_num)
        print("Fast mode: ", fast_mode)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
        num_layers = np.zeros(12, dtype=np.int32)

        model.eval()
        en = []
        if not args.mean_reciprocal_rank:
            total_flops, model_params_num = 0, 0
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(
                    batch_loader(batch_size, input_ids, label_ids, mask_ids)):

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():

                    # Get FLOPs at this batch
                    inputs = (input_ids_batch, label_ids_batch, mask_ids_batch, fast_mode)
                    flops, params = profile(model, inputs, verbose=False)
                    total_flops += flops
                    model_params_num = params

                    # inference
                    #loss, logits,num_layer, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, fast=fast_mode)
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, fast=fast_mode)
                    #num_layers+=num_layer
                logits = nn.Softmax(dim=1)(logits[-1])
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()
            #print(num_layers)
            print("Number of model parameters: {}".format(model_params_num))
            print("FLOPs per sample in average: {}".format(total_flops / float(instances_num)))

            # if is_test:
            # print("Confusion matrix:")
            # print(confusion)
            # print("Report precision, recall, and f1:")
            pm=0
            rm=0
            f1m=0
            for i in range(confusion.size()[0]):
                p = confusion[i,i].item()/(confusion[i,:].sum().item()+1e-7)
                pm+=p
                r = confusion[i,i].item()/(confusion[:,i].sum().item()+1e-7)
                rm+=r
                f1 = 2*p*r / ((p+r)+1e-7)
                f1m+=f1
            #print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            print(" {:.3f}, {:.3f}, {:.3f}".format(pm/9,rm/9,f1m/9))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))

            return correct / len(dataset)

    
    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path, False)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup, t_total=train_steps)

    # traning main part of model
    print("Start fine-tuning the backbone of the model.")
    start = time.time()
    total_loss = 0.
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, teacher=True)  # training
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            # niter = epoch * (instances_num / batch_size) + i
            # writer.add_scalars('OurmodelTrain_loss',  {'train_loss':loss.item()}, niter)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, backbone fine-tuning steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                              total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()

            optimizer.step()
            scheduler.step()
    save_model(model, args.output_teacher_model_path)
    '''
        result = evaluate(args, False, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_teacher_model_path)
        else:
            continue
    
    end = time.time()
    print('teacher:' + str(end - start))
    '''
    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation after bakbone fine-tuning.")
        model = load_model(model, args.output_teacher_model_path)
        print("Test on normal model")
        start = time.time()
        evaluate(args, True, False)
        #evaluate(args, True, args.fast_mode)
        end = time.time()
        print('teacher test:' + str(end - start))
    '''
    # Distillate subclassifiers
    print("Start self-distillation for student-classifiers.")

    print("Start self-distillation.")
    start=time.time()
    trainset_distillation = read_dataset(args.train_path,True)
    random.shuffle(trainset_distillation)
    instances_num_distillation = len(trainset_distillation)
    batch_size = args.batch_size

    input_ids_distillation = torch.LongTensor([example[0] for example in trainset_distillation])
    label_ids_distillation = torch.LongTensor([example[1] for example in trainset_distillation])
    mask_ids_distillation = torch.LongTensor([example[2] for example in trainset_distillation])
    train_steps = int(instances_num_distillation * args.distill_epochs_num / batch_size) + 1
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate*10, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps*args.warmup, t_total=train_steps)

    model = load_model(model, args.output_teacher_model_path)
    total_loss = 0.
    result = 0.0
    best_result = 0.0 
    for epoch in range(1, args.distill_epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids_distillation, label_ids_distillation, mask_ids_distillation)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch,teacher=False)  # distillation
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            niter = epoch * (instances_num_distillation / batch_size) + i
            writer.add_scalars('BERTTrain_loss',  {'train_loss':loss.item()}, niter)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, self-distillation steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()
            scheduler.step()
        #result = evaluate(args, False, args.fast_mode)
    save_model(model, args.output_student_model_path) 
    end=time.time()
    print('student:'+str(end-start))
    
    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation after self-distillation.")
        model = load_model(model, args.output_student_model_path)
        #for name,parameters in model.named_parameters():
        #    print(name,':',parameters.size())
        start=time.time()
        evaluate(args, True, args.fast_mode)
        #evaluate(args, True, False)
        end=time.time()
        print('student test fastmode:'+str(end-start))
    '''


if __name__ == "__main__":
    main()
