from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from nnet.util import *

import numpy as np
import torch
import math
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.nn.init as init
from numpy import random as nr
from operator import itemgetter
_BIG_NUMBER = 10. ** 6.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cat(l, dimension=-1):
    valid_l = l
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)

class BiLSTMTagger(nn.Module):

    #def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    def __init__(self, hps, *_):
        super(BiLSTMTagger, self).__init__()

        batch_size = hps['batch_size']
        lstm_hidden_dim = hps['sent_hdim']
        sent_embedding_dim_DEP = 2*hps['sent_edim'] + 16
        sent_embedding_dim_SRL = 3 * hps['sent_edim'] + 1 * hps['pos_edim'] + 16
        ## for the region mark
        role_embedding_dim = hps['role_edim']
        frame_embedding_dim = role_embedding_dim
        vocab_size = hps['vword']

        self.tagset_size = hps['vbio']
        self.pos_size = hps['vpos']
        self.dep_size = hps['vdep']
        self.frameset_size = hps['vframe']
        self.num_layers = hps['rec_layers']
        self.batch_size = batch_size
        self.hidden_dim = lstm_hidden_dim
        self.word_emb_dim = hps['sent_edim']
        self.specific_dep_size = hps['svdep']

        self.word_embeddings_SRL = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])
        self.pos_embeddings = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.pos_embeddings_DEP = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.p_lemma_embeddings = nn.Embedding(self.frameset_size, hps['sent_edim'])
        self.dep_embeddings = nn.Embedding(self.dep_size, self.pos_size)
        self.region_embeddings = nn.Embedding(2, 16)
        #self.lr_dep_embeddings = nn.Embedding(self.lr_dep_size, hps[])



        self.word_fixed_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.word_fixed_embeddings_DEP = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings_DEP.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))


        self.role_embeddings = nn.Embedding(self.tagset_size, role_embedding_dim)
        self.frame_embeddings = nn.Embedding(self.frameset_size, frame_embedding_dim)


        self.hidden2tag = nn.Linear(4*lstm_hidden_dim, 2*lstm_hidden_dim)
        self.MLP = nn.Linear(2*lstm_hidden_dim, self.dep_size)
        self.tag2hidden = nn.Linear(self.dep_size, self.pos_size)

        self.hidden2tag_spe = nn.Linear(2 * lstm_hidden_dim, 2 * lstm_hidden_dim)
        self.MLP_spe = nn.Linear(2 * lstm_hidden_dim, 4)
        self.tag2hidden_spe = nn.Linear(4, self.pos_size)

        #self.elmo_embeddings_0 = nn.Embedding(vocab_size, 1024)
        #self.elmo_embeddings_0.weight.data.copy_(torch.from_numpy(hps['elmo_embeddings_0']))

        #self.elmo_embeddings_1 = nn.Embedding(vocab_size, 1024)
        #self.elmo_embeddings_1.weight.data.copy_(torch.from_numpy(hps['elmo_embeddings_1']))


        self.elmo_emb_size = 200
        self.elmo_mlp_word = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
        self.elmo_word = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma_word = nn.Parameter(torch.ones(1))



        self.elmo_mlp = nn.Sequential(nn.Linear(2 * lstm_hidden_dim, self.elmo_emb_size), nn.ReLU())
        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.elmo_gamma = nn.Parameter(torch.ones(1))

        self.SRL_input_dropout = nn.Dropout(p=0.3)
        self.DEP_input_dropout = nn.Dropout(p=0.3)
        self.hidden_state_dropout = nn.Dropout(p=0.3)
        self.label_dropout = nn.Dropout(p=0.5)
        self.link_dropout = nn.Dropout(p=0.5)
        #self.use_dropout = nn.Dropout(p=0.2)



        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers = 1
        self.BiLSTM_0 = nn.LSTM(input_size=sent_embedding_dim_DEP , hidden_size=lstm_hidden_dim, batch_first=True,
                              bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_0.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_0.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_0.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_0.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_1 = nn.LSTM(input_size=lstm_hidden_dim * 2, hidden_size=lstm_hidden_dim, batch_first=True,
                                bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_1.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_1.all_weights[1][1])


        self.num_layers = 4
        self.BiLSTM_SRL = nn.LSTM(input_size=sent_embedding_dim_SRL + self.elmo_emb_size * 1 , hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])


        # non-linear map to role embedding
        self.role_map = nn.Linear(in_features=role_embedding_dim * 2, out_features=self.hidden_dim * 4)

        # Init hidden state
        self.hidden = self.init_hidden_spe()
        self.hidden_2 = self.init_hidden_spe()
        self.hidden_3 = self.init_hidden_spe()
        self.hidden_4 = self.init_hidden_share()
        self.ldims = lstm_hidden_dim

        self.hidLayerFOH = nn.Linear(self.ldims * 2, self.ldims)
        self.hidLayerFOM = nn.Linear(self.ldims * 2, self.ldims)
        self.W_R = nn.Parameter(torch.rand(lstm_hidden_dim + 1, 1 + lstm_hidden_dim))

        self.hidLayerFOH_tag = nn.Linear(self.ldims * 2, self.ldims)
        self.hidLayerFOM_tag = nn.Linear(self.ldims * 2, self.ldims)
        self.W_R_tag = nn.Parameter(torch.rand(lstm_hidden_dim + 1, self.dep_size*(1 + lstm_hidden_dim)))



        self.VR_word_embedding = nn.Parameter(
            torch.from_numpy(np.ones((1, self.word_emb_dim), dtype='float32')))

        self.VR_word_embedding_random = nn.Parameter(
            torch.from_numpy(np.ones((1, self.word_emb_dim), dtype='float32')))

        self.VR_POS_embedding = nn.Parameter(
            torch.from_numpy(np.ones((1, 16), dtype='float32')))

    def init_hidden_share(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(4 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(4 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

    def init_hidden_spe(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))


    def forward(self, sentence, p_sentence,  pos_tags, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx,  dep_tags,  dep_heads, targets, specific_dep_tags, specific_dep_relations, test=False):

        """
        elmo_embedding_0 = self.elmo_embeddings_0(sentence).view(self.batch_size, len(sentence[0]), 1024)
        elmo_embedding_1 = self.elmo_embeddings_1(sentence).view(self.batch_size, len(sentence[0]), 1024)
        w = F.softmax(self.elmo_word, dim=0)
        elmo_emb = self.elmo_gamma_word * (w[0] * elmo_embedding_0 + w[1] * elmo_embedding_1)
        elmo_emb_word = self.elmo_mlp_word(elmo_emb)
        """

        log(sentence)
        log(p_sentence)
        log(pos_tags)
        log(region_marks)
        log(sent_pred_lemmas_idx)

        embeds_DEP = self.word_embeddings_DEP(sentence)
        add_zero = torch.zeros((self.batch_size, 1, self.word_emb_dim)).to(device)
        embeds_DEP = embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        embeds_DEP_cat = torch.cat((self.VR_word_embedding_random + add_zero, embeds_DEP), 1)

        pos_embeds = self.pos_embeddings(pos_tags)
        add_zero = torch.zeros((self.batch_size, 1, 16)).to(device)
        pos_embeds_cat = torch.cat((self.VR_POS_embedding + add_zero, pos_embeds), 1)

        fixed_embeds_DEP = self.word_fixed_embeddings(p_sentence)
        fixed_embeds_DEP = fixed_embeds_DEP.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        add_zero = torch.zeros((self.batch_size, 1, self.word_emb_dim)).to(device)
        fixed_embeds_DEP_cat = torch.cat((self.VR_word_embedding + add_zero, fixed_embeds_DEP), 1)
        embeds_forDEP = torch.cat((embeds_DEP_cat, fixed_embeds_DEP_cat, pos_embeds_cat), 2)
        embeds_forDEP = self.DEP_input_dropout(embeds_forDEP)


        # first layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds_forDEP, lengths + 1)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden = self.BiLSTM_0(embeds_sort, self.hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_0 = hidden_states[unsort_idx]

        # second_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states_0, lengths + 1)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_2 = self.BiLSTM_1(embeds_sort, self.hidden_2)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states_1 = hidden_states[unsort_idx]

        ##########################################
        Head_hidden = F.relu(self.hidLayerFOH(hidden_states_1))
        Dependent_hidden = F.relu(self.hidLayerFOM(hidden_states_1))

        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Head_hidden = torch.cat((Head_hidden, Variable(bias_one)), 2)

        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden = torch.cat((Dependent_hidden, Variable(bias_one)), 2)

        left_part = torch.mm(Dependent_hidden.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1), -1)
        Head_hidden = Head_hidden.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space = torch.bmm(left_part, Head_hidden).view(
            (len(sentence[0]) + 1) * self.batch_size, len(sentence[0]) + 1)

        heads = np.argmax(tag_space.cpu().data.numpy(), axis=1)

        nums = 0.0
        wrong_nums = 0.0
        for a, b in zip(heads, dep_heads.flatten()):
            if b == -1:
                continue
            nums += 1
            if a != b:
                wrong_nums += 1

        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        DEPloss = loss_function(tag_space, torch.from_numpy(dep_heads).to(device).view(-1))

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        Head_hidden_tag = F.relu(self.hidLayerFOH_tag(hidden_states_1))
        Dependent_hidden_tag = F.relu(self.hidLayerFOM_tag(hidden_states_1))

        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Head_hidden_tag = torch.cat((Head_hidden_tag, Variable(bias_one)), 2)

        bias_one = torch.ones((self.batch_size, len(sentence[0]) + 1, 1)).to(device)
        Dependent_hidden_tag = torch.cat((Dependent_hidden_tag, Variable(bias_one)), 2)

        left_part = torch.mm(Dependent_hidden_tag.view(self.batch_size * (len(sentence[0]) + 1), -1), self.W_R_tag)
        left_part = left_part.view(self.batch_size, (len(sentence[0]) + 1)* self.dep_size, -1)
        Head_hidden_tag = Head_hidden_tag.view(self.batch_size, (len(sentence[0]) + 1), -1).transpose(1, 2)
        tag_space_tag = torch.bmm(left_part, Head_hidden_tag).view(
            (len(sentence[0]) + 1) * self.batch_size, self.dep_size, len(sentence[0]) + 1).transpose(1, 2)

        tag_space_tag = tag_space_tag[np.arange(0, (len(sentence[0]) + 1) * self.batch_size), dep_heads.flatten()]
        tag_space_tag = tag_space_tag.view((len(sentence[0]) + 1) * self.batch_size, -1)
        heads_tag = np.argmax(tag_space_tag.cpu().data.numpy(), axis=1)

        nums_tag = 0.0
        wrong_nums_tag = 0.0
        for a, b in zip(heads_tag, dep_tags.view(-1).cpu().data.numpy()):
            if b == -1 or b==0:
                continue
            nums_tag += 1
            if a != b:
                wrong_nums_tag += 1

        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        DEPloss_tag = loss_function(tag_space_tag, dep_tags.view(-1))

        h_layer_0 = hidden_states_0[:, 1:]  # .detach()
        h_layer_1 = hidden_states_1[:, 1:]  # .detach()

        w = F.softmax(self.elmo_w, dim=0)
        SRL_composer = self.elmo_gamma * (w[0] * h_layer_0 + w[1] * h_layer_1)
        SRL_composer = self.elmo_mlp(SRL_composer)


        fixed_embeds = self.word_fixed_embeddings(p_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)
        embeds_SRL = self.word_embeddings_SRL(sentence)
        embeds_SRL = embeds_SRL.view(self.batch_size, len(sentence[0]), self.word_emb_dim)

        region_marks = self.region_embeddings(region_marks).view(self.batch_size, len(sentence[0]), 16)
        SRL_hidden_states = torch.cat((embeds_SRL,  fixed_embeds, sent_pred_lemmas_embeds, pos_embeds, region_marks,
                                      SRL_composer), 2)
        SRL_hidden_states = self.SRL_input_dropout(SRL_hidden_states)


        # SRL layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(SRL_hidden_states, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_4 = self.BiLSTM_SRL(embeds_sort, self.hidden_4)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]
        hidden_states = self.hidden_state_dropout(hidden_states)


        # B * H
        hidden_states_3 = hidden_states
        predicate_embeds = hidden_states_3[np.arange(0, hidden_states_3.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = torch.zeros(hidden_states_3.size()[1], hidden_states_3.size()[0], hidden_states_3.size()[2]).to(device)
        predicate_embeds = added_embeds + predicate_embeds
        # B * T * H
        predicate_embeds = predicate_embeds.transpose(0, 1)
        hidden_states = torch.cat((hidden_states_3, predicate_embeds), 2)
        # print(hidden_states)
        # non-linear map and rectify the roles' embeddings
        # roles = Variable(torch.from_numpy(np.arange(0, self.tagset_size)))

        # B * roles
        # log(local_roles_voc)
        # log(frames)

        # B * roles * h
        role_embeds = self.role_embeddings(local_roles_voc)
        frame_embeds = self.frame_embeddings(frames)

        role_embeds = torch.cat((role_embeds, frame_embeds), 2)
        mapped_roles = F.relu(self.role_map(role_embeds))
        mapped_roles = torch.transpose(mapped_roles, 1, 2)

        # b, times, roles
        tag_space = torch.matmul(hidden_states, mapped_roles)
        #tag_space = hidden_states.mm(mapped_roles)



        # b, roles
        #sub = torch.div(torch.add(local_roles_mask, -1.0), _BIG_NUMBER)
        sub = torch.add(local_roles_mask, -1.0) * _BIG_NUMBER
        sub = torch.FloatTensor(sub.cpu().numpy()).to(device)
        # b, roles, times
        tag_space = torch.transpose(tag_space, 0, 1)
        tag_space += sub
        # b, T, roles
        tag_space = torch.transpose(tag_space, 0, 1)
        tag_space = tag_space.view(len(sentence[0])*self.batch_size, -1)

        SRLprobs = F.softmax(tag_space, dim=1)



        targets = targets.view(-1)

        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        SRLloss = loss_function(tag_space, targets)


        loss = SRLloss + DEPloss+ DEPloss_tag
        return SRLloss, DEPloss, DEPloss_tag, loss, SRLprobs, wrong_nums, nums, wrong_nums, nums,  \
               wrong_nums, nums, nums,\
               wrong_nums_tag, nums_tag, nums_tag

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx