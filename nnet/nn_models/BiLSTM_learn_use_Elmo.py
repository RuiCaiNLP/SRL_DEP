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
_BIG_NUMBER = 10. ** 6.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BiLSTMTagger(nn.Module):

    #def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    def __init__(self, hps, *_):
        super(BiLSTMTagger, self).__init__()

        batch_size = hps['batch_size']
        lstm_hidden_dim = hps['sent_hdim']
        sent_embedding_dim = 3*hps['sent_edim'] + 1*hps['pos_edim']
        ## for the region mark
        sent_embedding_dim += 1
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

        self.word_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.pos_embeddings = nn.Embedding(self.pos_size, hps['pos_edim'])
        self.dep_embeddings = nn.Embedding(self.dep_size, hps['pos_edim'])
        self.p_lemma_embeddings = nn.Embedding(self.frameset_size, hps['sent_edim'])
        #self.lr_dep_embeddings = nn.Embedding(self.lr_dep_size, hps[])

        self.word_fixed_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.elmo_embeddings_0 = nn.Embedding(vocab_size, 1024)
        self.elmo_embeddings_0.weight.data.copy_(torch.from_numpy(hps['elmo_embeddings_0']))
        #self.elmo_embeddings_0.weight.requires_grad_(False)

        self.elmo_embeddings_1 = nn.Embedding(vocab_size, 1024)
        self.elmo_embeddings_1.weight.data.copy_(torch.from_numpy(hps['elmo_embeddings_1']))
        #self.elmo_embeddings_1.weight.requires_grad_(False)

        self.role_embeddings = nn.Embedding(self.tagset_size, role_embedding_dim)
        self.frame_embeddings = nn.Embedding(self.frameset_size, frame_embedding_dim)

        self.hidden2tag = nn.Linear(200, 200)
        self.MLP = nn.Linear(200, self.dep_size)

        self.tag2hidden = nn.Linear(self.dep_size, self.pos_size)

        self.hidden2tag_spe = nn.Linear(100, 100)
        self.MLP_spe = nn.Linear(100, 4)
        self.Link2hidden = nn.Linear(4, self.pos_size)

        self.word_emb_dropout = nn.Dropout(p=0.3)
        self.hidden_state_dropout = nn.Dropout(p=0.3)
        self.label_dropout = nn.Dropout(p=0.5)
        self.link_dropout = nn.Dropout(p=0.5)
        #self.use_dropout = nn.Dropout(p=0.2)

        self.elmo_w = nn.Parameter(torch.Tensor([0.5, 0.5]))

        self.elmo_gamma = nn.Parameter(torch.ones(1))
        self.elmo_project = nn.Linear(1024, 100)




        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers = 2
        self.BiLSTM_share = nn.LSTM(input_size=sent_embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True,
                              bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_share.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_share.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_share.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_share.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_Spe = nn.LSTM(input_size=lstm_hidden_dim * 2, hidden_size=lstm_hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_Spe.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_Spe.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_Spe.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_Spe.all_weights[1][1])

        self.num_layers = 1
        self.BiLSTM_SRL = nn.LSTM(input_size=lstm_hidden_dim * 2 + 2 * self.pos_size , hidden_size=lstm_hidden_dim, batch_first=True,
                                    bidirectional=True, num_layers=self.num_layers)

        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_SRL.all_weights[1][1])


        # non-linear map to role embedding
        self.role_map = nn.Linear(in_features=role_embedding_dim * 2, out_features=self.hidden_dim * 4)

        # Init hidden state
        self.hidden = self.init_hidden_share()
        self.hidden_2 = self.init_hidden_spe()
        self.hidden_3 = self.init_hidden_spe()
        self.hidden_4 = self.init_hidden_spe()

    def init_hidden_share(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device),
                torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False).to(device))

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



        embeds = self.word_embeddings(sentence)
        embeds = embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        pos_embeds = self.pos_embeddings(pos_tags)
        fixed_embeds = self.word_fixed_embeddings(sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)

        elmo_embedding_0 = self.elmo_embeddings_0(sentence).view(self.batch_size, len(sentence[0]), 1024)
        elmo_embedding_1 = self.elmo_embeddings_1(sentence).view(self.batch_size, len(sentence[0]), 1024)
        w = F.softmax(self.elmo_w, dim=0)
        elmo_emb = self.elmo_gamma * (w[0] * elmo_embedding_0 + w[1]* elmo_embedding_1)
        elmo_emb_weighted = F.relu(self.elmo_project(elmo_emb), inplace=True)

        region_marks = region_marks.view(self.batch_size, len(sentence[0]), 1)
        embeds = torch.cat((embeds, fixed_embeds,  pos_embeds,  sent_pred_lemmas_embeds, region_marks), 2)
        #embeds = torch.cat((embeds, fixed_embeds, pos_embeds, region_marks), 2)
        embeds = self.word_emb_dropout(embeds)

        # share_layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(embeds, lengths)

        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden = self.BiLSTM_share(embeds_sort, self.hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        #hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]

        forward_h, backward_h = torch.split(hidden_states, self.hidden_dim, 2)
        forward_e = forward_h[:, :, :50]
        backward_e = backward_h[:, :, :50]
        bf_e = torch.cat((forward_e, backward_e), 2)

        predicate_embeds = bf_e[np.arange(0, bf_e.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = torch.zeros(bf_e.size()[1], bf_e.size()[0], bf_e.size()[2]).to(device)
        concat_embeds = added_embeds + predicate_embeds
        bf_e = torch.cat((bf_e, concat_embeds.transpose(0, 1)), 2)
        dep_tag_space = self.MLP(self.label_dropout(F.tanh(self.hidden2tag(bf_e)))).view(
            len(sentence[0]) * self.batch_size, -1)


        # Spe layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_2 = self.BiLSTM_Spe(embeds_sort, self.hidden_2)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states[unsort_idx]

        forward_h, backward_h = torch.split(hidden_states, self.hidden_dim, 2)
        forward_e = forward_h[:, :, :50]
        backward_e = backward_h[:, :, :50]
        bf_e = torch.cat((forward_e, backward_e), 2)

        dep_tag_space_spe = self.MLP_spe(self.link_dropout(F.tanh(self.hidden2tag_spe(bf_e)))).view(
            len(sentence[0]) * self.batch_size, -1)

        #TagProbs = torch.FloatTensor(F.softmax(dep_tag_space, dim=1).view(self.batch_size, len(sentence[0]), -1).cpu().data.numpy()).to(device)
        #LinkProbs = torch.FloatTensor(F.softmax(dep_tag_space_spe, dim=1).view(self.batch_size, len(sentence[0]), -1).cpu().data.numpy()).to(device)

        TagProbs = F.softmax(dep_tag_space, dim=1).view(self.batch_size, len(sentence[0]), -1)
        LinkProbs = F.softmax(dep_tag_space_spe, dim=1).view(self.batch_size, len(sentence[0]), -1)
        h1 = F.relu(self.tag2hidden(TagProbs), inplace=True)
        h2 = F.relu(self.Link2hidden(LinkProbs), inplace=True)
        #H_use = self.use_dropout(torch.cat((h1, h2), 2))
        hidden_states = torch.cat((hidden_states, h1, h2), 2)

        # SRL layer
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(hidden_states, lengths)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort.cpu().numpy(), batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, self.hidden_3 = self.BiLSTM_SRL(embeds_sort, self.hidden_3)
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

        #+++++++++++++++++++++++
        wrong_l_nums = 0.0
        all_l_nums = 0.0

        right_noNull_predict = 0.0
        noNull_predict = 0.0
        noNUll_truth = 0.0
        dep_labels = np.argmax(dep_tag_space.cpu().data.numpy(), axis=1)
        for predict_l, gold_l in zip(dep_labels, dep_tags.cpu().view(-1).data.numpy()):
            if predict_l >1:
                noNull_predict += 1
            if gold_l != 0:
                all_l_nums += 1
                if gold_l != 1:
                    noNUll_truth += 1
                    if gold_l == predict_l:
                        right_noNull_predict += 1
            if predict_l != gold_l and gold_l != 0:
                wrong_l_nums += 1


        #+++++++++++++++++++++++
        wrong_l_nums_spe = 0.0
        all_l_nums_spe = 0.0

        right_noNull_predict_spe = 0.0
        noNull_predict_spe = 0.0
        noNUll_truth_spe = 0.0
        spe_dep_labels = np.argmax(dep_tag_space_spe.cpu().data.numpy(), axis=1)
        for predict_l, gold_l in zip(spe_dep_labels, specific_dep_relations.cpu().view(-1).data.numpy()):
            if predict_l != 0 and predict_l != 3:
                noNull_predict_spe += 1
            if gold_l != 0:
                all_l_nums_spe += 1
                if gold_l != 3:
                    noNUll_truth_spe += 1
                    if gold_l == predict_l:
                        right_noNull_predict_spe += 1
            if predict_l != gold_l and gold_l != 0:
                wrong_l_nums_spe += 1

        #loss_function = nn.NLLLoss(ignore_index=0)
        targets = targets.view(-1)
        #tag_scores = F.log_softmax(tag_space)
        #loss = loss_function(tag_scores, targets)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        SRLloss = loss_function(tag_space, targets)
        DEPloss = loss_function(dep_tag_space, dep_tags.view(-1))
        SPEDEPloss = loss_function(dep_tag_space_spe, specific_dep_relations.view(-1))


        #weight = float(SRLloss.cpu().data.numpy())
        #if weight > 0.1:
        #    weight = 0.1
        #p = nr.rand()
        #if p<0.2:
        #    loss = SRLloss + DEPloss + SPEDEPloss
        #else:
        #    loss = SRLloss
        loss =SRLloss # 0.1*DEPloss + 0.1*SPEDEPloss
        return SRLloss, DEPloss, SPEDEPloss, loss, SRLprobs, wrong_l_nums, all_l_nums, wrong_l_nums_spe, all_l_nums_spe,  \
               right_noNull_predict, noNull_predict, noNUll_truth,\
               right_noNull_predict_spe, noNull_predict_spe, noNUll_truth_spe

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx


