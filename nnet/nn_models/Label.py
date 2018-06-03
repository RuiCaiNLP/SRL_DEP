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
_BIG_NUMBER = 10. ** 6.


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
        self.p_lemma_embeddings = nn.Embedding(self.frameset_size, hps['sent_edim'])
        #self.lr_dep_embeddings = nn.Embedding(self.lr_dep_size, hps[])

        self.word_fixed_embeddings = nn.Embedding(vocab_size, hps['sent_edim'])
        self.word_fixed_embeddings.weight.data.copy_(torch.from_numpy(hps['word_embeddings']))

        self.role_embeddings = nn.Embedding(self.tagset_size, role_embedding_dim)
        self.frame_embeddings = nn.Embedding(self.frameset_size, frame_embedding_dim)

        self.hidden2tag_M = nn.Linear(100, 200)
        self.hidden2tag_H = nn.Linear(100, 200)
        self.MLP = nn.Linear(200, self.dep_size)


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
        self.BiLSTM_SRL = nn.LSTM(input_size=lstm_hidden_dim * 2, hidden_size=lstm_hidden_dim, batch_first=True,
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
        return (torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False),
                torch.zeros(2 * 2, self.batch_size, self.hidden_dim, requires_grad=False))

    def init_hidden_spe(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False),
                torch.zeros(1 * 2, self.batch_size, self.hidden_dim, requires_grad=False))


    def forward(self, sentence, p_sentence,  pos_tags, lengths, target_idx_in, region_marks,
                local_roles_voc, frames, local_roles_mask,
                sent_pred_lemmas_idx,  dep_tags,  dep_heads, targets, specific_dep_tags, specific_dep_relations, test=False):



        embeds = self.word_embeddings(sentence)
        embeds = embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        pos_embeds = self.pos_embeddings(pos_tags)
        fixed_embeds = self.word_fixed_embeddings(p_sentence)
        fixed_embeds = fixed_embeds.view(self.batch_size, len(sentence[0]), self.word_emb_dim)
        sent_pred_lemmas_embeds = self.p_lemma_embeddings(sent_pred_lemmas_idx)


        region_marks = region_marks.view(self.batch_size, len(sentence[0]), 1)
        embeds = torch.cat((embeds, fixed_embeds, pos_embeds,  sent_pred_lemmas_embeds, region_marks), 2)



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
        concat_embeds = torch.zeros(bf_e.size()[0], bf_e.size()[1], bf_e.size()[2])
        for i in range(bf_e.size()[0]):
            for j in range(bf_e.size()[1]):
                if dep_heads[i][j] > 0:
                    concat_embeds[i, j] = bf_e[i, dep_heads[i][j] - 1]
        dep_tag_space = self.MLP(F.tanh(self.hidden2tag_M(bf_e) + self.hidden2tag_H(concat_embeds))).view(
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



        # B * H
        hidden_states_3 = hidden_states
        predicate_embeds = hidden_states_3[np.arange(0, hidden_states_3.size()[0]), target_idx_in]
        # T * B * H
        added_embeds = Variable(torch.zeros(hidden_states_3.size()[1], hidden_states_3.size()[0], hidden_states_3.size()[2]))
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
        sub = torch.FloatTensor(sub.numpy())
        # b, roles, times
        tag_space = torch.transpose(tag_space, 0, 1)
        tag_space += sub
        # b, T, roles
        tag_space = torch.transpose(tag_space, 0, 1)

        tag_space = tag_space.view(len(sentence[0])*self.batch_size, -1)

        SRLprobs = F.softmax(tag_space, dim=1)
        wrong_l_nums = 0.0
        all_l_nums = 0.0


        dep_labels = np.argmax(dep_tag_space.data.numpy(), axis=1)
        for predict_l, gold_l in zip(dep_labels, dep_tags.view(-1).data.numpy()):

            if gold_l != 0:
                all_l_nums += 1
            if predict_l != gold_l and gold_l != 0:
                wrong_l_nums += 1


        targets = targets.view(-1)
        #tag_scores = F.log_softmax(tag_space)
        #loss = loss_function(tag_scores, targets)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        SRLloss = loss_function(tag_space, targets)
        DEPloss = loss_function(dep_tag_space, dep_tags.view(-1))
        #SPEDEPloss = loss_function(dep_tag_space_spe, specific_dep_relations.view(-1))


        loss = SRLloss + 0.1*DEPloss #+ 0.1*SPEDEPloss
        return SRLloss, DEPloss, DEPloss, loss, SRLprobs, wrong_l_nums, all_l_nums, 1, 1

    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx
