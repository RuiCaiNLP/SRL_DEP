import torch
import torch.tensor
from nnet.util import *
import torch.autograd
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_
from torch import optim
import time
import random
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_local_voc(labels):
    return {i: label for i, label in enumerate(labels)}


def train(model, train_set, dev_set, test_set, epochs, converter, dbg_print_rate, params_path):
    idx = 0
    sample_count = 0.0
    best_F1 = -0.1
    #optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002, betas=(0.9, 0.9), eps=1e-12)
    #log(optimizer.param_groups[0]['lr'])
    #optimizer.param_groups[0]['lr'] = 0.001

    """
    Best_BiLSTM_0_data = []
    for weight in model.BiLSTM_0.parameters():
        Best_BiLSTM_0_data.append(weight.data.clone())
    Best_BiLSTM_1_data = []
    for weight in model.BiLSTM_1.parameters():
        Best_BiLSTM_1_data.append(weight.data.clone())
    best_word_embeddings_DEP = model.word_embeddings_DEP.weight.data.clone()
    best_pos_embeddings_DEP = model.pos_embeddings_DEP.weight.data.clone()
    best_word_fixed_embeddings_DEP = model.word_fixed_embeddings_DEP.weight.data.clone()
    best_hidden2tag = model.hidden2tag.weight.data.clone()
    best_MLP = model.MLP.weight.data.clone()
    """
    Best_DEP_score = -0.1

    random.seed(1234)
    for e in range(epochs):
        tic = time.time()
        dataset = [batch for batch in train_set.batches()]
        init_dataset = [batch for batch in dataset]
        #random.shuffle(dataset)
        dataset_len = len(dataset)
        for batch in dataset:

            batch_idx = init_dataset.index(batch)
            sample_count += len(batch)

            model.zero_grad()
            optimizer.zero_grad()
            model.train()
            record_ids, batch = zip(*batch)
            model_input = converter(batch)

            model.hidden = model.init_hidden_spe()
            #model.hidden_0 = model.init_hidden_spe()
            model.hidden_2 = model.init_hidden_spe()
            model.hidden_3 = model.init_hidden_spe()
            model.hidden_4 = model.init_hidden_share()

            sentence = model_input[0]
            p_sentence = model_input[1]

            sentence_in = torch.from_numpy(sentence).to(device)
            p_sentence_in = torch.from_numpy(p_sentence).to(device)
            #log(sentence_in)
            #log(p_sentence_in)
            #sentence_in.requires_grad_(False)
            #p_sentence_in.requires_grad_(False)

            pos_tags = model_input[2]
            pos_tags_in = torch.from_numpy(pos_tags).to(device)
            #pos_tags_in.requires_grad_(False)

            sen_lengths = model_input[3].sum(axis=1)

            target_idx_in = model_input[4]

            frames = model_input[5]
            frames_in = torch.from_numpy(frames).to(device)
            #frames_in.requires_grad_(False)

            local_roles_voc = model_input[6]
            local_roles_voc_in = torch.from_numpy(local_roles_voc).to(device)
            #local_roles_voc_in.requires_grad_(False)

            local_roles_mask = model_input[7]
            local_roles_mask_in = torch.from_numpy(local_roles_mask).to(device)
            #local_roles_mask_in.requires_grad_(False)

            region_mark = model_input[9]

            # region_mark_in = Variable(torch.LongTensor(region_mark))
            region_mark_in = torch.from_numpy(region_mark).to(device)
            #region_mark_in.requires_grad_(False)

            sent_pred_lemmas_idx = model_input[10]
            sent_pred_lemmas_idx_in = torch.from_numpy(sent_pred_lemmas_idx).to(device)
            #sent_pred_lemmas_idx_in.requires_grad_(False)

            dep_tags = model_input[11]
            dep_tags_in = torch.from_numpy(dep_tags).to(device)


            dep_heads = model_input[12]


            #root_dep_tags = model_input[12]
            #root_dep_tags_in = Variable(torch.from_numpy(root_dep_tags), requires_grad=False)

            tags = model_input[13]
            targets = torch.tensor(tags).to(device)

            specific_dep_tags = model_input[14]
            specific_dep_tags_in = torch.from_numpy(specific_dep_tags).to(device)

            specific_dep_relations = model_input[15]
            specific_dep_relations_in = torch.from_numpy(specific_dep_relations).to(device)


            #log(dep_tags_in)
            #log(specific_dep_relations)
            SRLloss, DEPloss, SPEDEPloss, loss, SRLprobs, wrong_l_nums, all_l_nums, spe_wrong_l_nums, spe_all_l_nums, \
            right_noNull_predict, noNull_predict, noNUll_truth, \
            right_noNull_predict_spe, noNull_predict_spe, noNUll_truth_spe\
                = model(sentence_in, p_sentence_in, pos_tags_in, sen_lengths, target_idx_in, region_mark_in,
                        local_roles_voc_in,
                        frames_in, local_roles_mask_in, sent_pred_lemmas_idx_in, dep_tags_in, dep_heads,
                        targets, specific_dep_tags_in, specific_dep_relations_in)




            idx += 1
            #Final_loss = SRLloss + 0.5/(1 + 0.3 *(e-1)) * (DEPloss + SPEDEPloss)
            #if batch_idx < dataset_len * 0.9:
            #    Final_loss = SRLloss + 0.5 * (DEPloss + SPEDEPloss)
            #else:
            #    Final_loss = SRLloss

            Final_loss = SRLloss
            Final_loss.backward()
            #clip_grad_norm_(parameters=model.hidden2tag_M.parameters(), max_norm=norm)
            #clip_grad_norm_(parameters=model.hidden2tag_H.parameters(), max_norm=norm)
            #clip_grad_value_(parameters=model.parameters(), clip_value=3)
            #DEPloss.backward()
            optimizer.step()


            #if idx % 10000 == 0:
            #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.75

            if idx % 100 ==0:
                log(idx)
                log("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                log('SRLloss')
                log(SRLloss)
                log("DEPloss")
                log(DEPloss)
                log("SPEDEPloss")
                log(SPEDEPloss)
                log("sum")
                log(loss)

                del SRLloss
                del DEPloss
                del SPEDEPloss
                del loss
                del SRLprobs
                del model.hidden
                del model.hidden_2
                del model.hidden_3
                del model.hidden_4

            if idx % dbg_print_rate == 0:
                log('[epoch %i, %i * %i] ' %
                    (e, idx, len(batch)))

                log("start test...")
                losses, errors, errors_w, NonNullPredicts, right_NonNullPredicts, NonNullTruths = 0., 0, 0., 0., 0., 0.
                total_labels_num = 0.0
                wrong_labels_num = 0.0
                spe_total_labels_num = 0.0
                spe_wrong_labels_num = 0.0

                right_noNull_predict =0.0
                noNull_predict = 0
                noNUll_truth = 0.0

                right_noNull_predict_spe = 0
                noNull_predict_spe = 0
                noNUll_truth_spe = 0


                Dep_count_num = [0.0] * 100
                Dep_NoNull_Truth = [0.0] * 100
                Dep_NoNull_Predict = [0.0] * 100
                Dep_Right_NoNull_Predict = [0.0] * 100

                Dep_P = [0.0] * 100
                Dep_R = [0.0] * 100
                Dep_F = [0.0] * 100

                log('now dev test')
                index = 0

                model.eval()
                with torch.no_grad():
                    for batch in dev_set.batches():
                        index += 1
                        # loss, e, e_w, NonNullPredict, right_NonNullPredict, NonNullTruth = self.error_computer.compute(model, batch)
                        errors, errors_w = 0, 0.0
                        NonNullPredict = 0
                        NonNullTruth = 0
                        right_NonNullPredict = 0

                        model.zero_grad()
                        optimizer.zero_grad()

                        record_ids, batch = zip(*batch)
                        model_input = converter(batch)
                        model.hidden = model.init_hidden_spe()
                        #model.hidden_0 = model.init_hidden_spe()
                        model.hidden_2 = model.init_hidden_spe()
                        model.hidden_3 = model.init_hidden_spe()
                        model.hidden_4 = model.init_hidden_share()


                        sentence = model_input[0]
                        p_sentence = model_input[1]

                        sentence_in = torch.from_numpy(sentence).to(device)
                        p_sentence_in = torch.from_numpy(p_sentence).to(device)
                        sentence_in.requires_grad_(False)
                        p_sentence_in.requires_grad_(False)

                        pos_tags = model_input[2]
                        pos_tags_in = torch.from_numpy(pos_tags).to(device)
                        pos_tags_in.requires_grad_(False)

                        sen_lengths = model_input[3].sum(axis=1)

                        target_idx_in = model_input[4]

                        frames = model_input[5]
                        frames_in = torch.from_numpy(frames).to(device)
                        frames_in.requires_grad_(False)

                        local_roles_voc = model_input[6]
                        local_roles_voc_in = torch.from_numpy(local_roles_voc).to(device)
                        local_roles_voc_in.requires_grad_(False)

                        local_roles_mask = model_input[7]
                        local_roles_mask_in = torch.from_numpy(local_roles_mask).to(device)
                        local_roles_mask_in.requires_grad_(False)

                        region_mark = model_input[9]

                        # region_mark_in = Variable(torch.LongTensor(region_mark))
                        region_mark_in = torch.from_numpy(region_mark).to(device)
                        region_mark_in.requires_grad_(False)

                        sent_pred_lemmas_idx = model_input[10]
                        sent_pred_lemmas_idx_in = torch.from_numpy(sent_pred_lemmas_idx).to(device)
                        sent_pred_lemmas_idx_in.requires_grad_(False)

                        dep_tags = model_input[11]
                        dep_tags_in = torch.from_numpy(dep_tags).to(device)

                        dep_heads = model_input[12]

                        # root_dep_tags = model_input[12]
                        # root_dep_tags_in = Variable(torch.from_numpy(root_dep_tags), requires_grad=False)

                        tags = model_input[13]
                        targets = torch.tensor(tags).to(device)

                        specific_dep_tags = model_input[14]
                        specific_dep_tags_in = torch.from_numpy(specific_dep_tags).to(device)

                        specific_dep_relations = model_input[15]
                        specific_dep_relations_in = Variable(torch.from_numpy(specific_dep_relations)).to(device)

                        SRLloss, DEPloss, SPEDEPloss, loss, SRLprobs, wrong_l_nums, all_l_nums, spe_wrong_l_nums, spe_all_l_nums, \
                        right_noNull_predict_b, noNull_predict_b, noNUll_truth_b, \
                        right_noNull_predict_spe_b, noNull_predict_spe_b, noNUll_truth_spe_b\
                            = model(sentence_in, p_sentence_in, pos_tags_in, sen_lengths, target_idx_in, region_mark_in,
                                    local_roles_voc_in,
                                    frames_in, local_roles_mask_in, sent_pred_lemmas_idx_in, dep_tags_in, dep_heads,
                                    targets, specific_dep_tags_in, specific_dep_relations_in, True)

                        labels = np.argmax(SRLprobs.cpu().data.numpy(), axis=1)
                        labels = np.reshape(labels, sentence.shape)
                        wrong_labels_num += wrong_l_nums
                        total_labels_num += all_l_nums
                        spe_wrong_labels_num += spe_wrong_l_nums
                        spe_total_labels_num += spe_all_l_nums

                        right_noNull_predict += right_noNull_predict_b
                        noNull_predict += noNull_predict_b
                        noNUll_truth += noNUll_truth_b
                        right_noNull_predict_spe += right_noNull_predict_spe_b
                        noNull_predict_spe += noNull_predict_spe_b
                        noNUll_truth_spe += noNUll_truth_spe_b

                        for i, sent_labels in enumerate(labels):
                            labels_voc = batch[i][-4]
                            local_voc = make_local_voc(labels_voc)
                            for j in range(len(labels[i])):
                                best = local_voc[labels[i][j]]
                                true = local_voc[tags[i][j]]
                                
                                if true != '<pad>':
                                    Dep_count_num[dep_tags_in[i][j]] += 1
                                if true != '<pad>' and true != 'O':
                                    NonNullTruth += 1
                                    Dep_NoNull_Truth[dep_tags_in[i][j]] += 1
                                if true != best:
                                    errors += 1
                                if best != '<pad>' and best != 'O' and true != '<pad>':
                                    NonNullPredict += 1
                                    Dep_NoNull_Predict[dep_tags_in[i][j]] += 1
                                    if true == best:
                                        right_NonNullPredict += 1
                                        Dep_Right_NoNull_Predict[dep_tags_in[i][j]] += 1

                        NonNullPredicts += NonNullPredict
                        right_NonNullPredicts += right_NonNullPredict
                        NonNullTruths += NonNullTruth

                        del SRLloss
                        del DEPloss
                        del SPEDEPloss
                        del loss
                        del SRLprobs
                        del model.hidden
                        del model.hidden_2
                        del model.hidden_3
                        del model.hidden_4

                for i in range(len(Dep_P)):
                    Dep_P[i] = Dep_Right_NoNull_Predict[i] / (Dep_NoNull_Predict[i] + 0.0001)
                    Dep_R[i] = Dep_Right_NoNull_Predict[i] / (Dep_NoNull_Truth[i] + 0.0001)
                    Dep_F[i] = 2 * Dep_P[i] * Dep_R[i] / (Dep_P[i] + Dep_R[i] + 0.0001)
                    if int(Dep_count_num[i]) > 0:
                        log(str(int(Dep_count_num[i])) + '\t' + str(Dep_P[i]) + '\t' + str(Dep_R[i]) + '\t' + str(Dep_F[i]))
                Predicat_num = 6300
                P = (right_NonNullPredicts + Predicat_num) / (NonNullPredicts + Predicat_num)
                R = (right_NonNullPredicts + Predicat_num) / (NonNullTruths + Predicat_num)
                F1 = 2 * P * R / (P + R)
                log(right_NonNullPredicts)
                log(NonNullPredicts)
                log(NonNullTruths)
                log('Precision: ' + str(P), 'recall: ' + str(R), 'F1: ' + str(F1))
                P = (right_NonNullPredicts) / (NonNullPredicts + 1)
                R = (right_NonNullPredicts) / (NonNullTruths)
                F1 = 2 * P * R / (P + R + 0.0001)
                log('Precision: ' + str(P), 'recall: ' + str(R), 'F1: ' + str(F1))
                log('Best F1: ' + str(best_F1))
                if F1 > best_F1:
                    best_F1 = F1
                    torch.save(model.state_dict(), params_path)
                    log('New best, model saved')

                P = right_noNull_predict / (noNull_predict + 0.0001)
                R = right_noNull_predict / (noNUll_truth + 0.0001)
                F_label = 2 * P * R / (P + R + 0.0001)
                log('Label Precision: P, R, F:' + str(P) + ' ' + str(R) + ' ' + str(F_label))

                P = right_noNull_predict_spe / (noNull_predict_spe + 0.0001)
                R = right_noNull_predict_spe / (noNUll_truth_spe + 0.0001)
                F_link = 2 * P * R / (P + R + 0.0001)
                log('Label Precision: P, R, F:' + str(P) + ' ' + str(R) + ' ' + str(F_link))
                if F_label> Best_DEP_score and e < 20 and False:
                    Best_DEP_score = F_label
                    log('New Dep best:' + str(Best_DEP_score))
                    for best_weight_i, weight_i in zip(Best_BiLSTM_0_data, model.BiLSTM_0.parameters()):
                        best_weight_i.copy_(weight_i.data)
                    for best_weight_i, weight_i in zip(Best_BiLSTM_1_data, model.BiLSTM_1.parameters()):
                        best_weight_i.copy_(weight_i.data)
                    best_word_embeddings_DEP = model.word_embeddings_DEP.weight.data.clone()
                    best_pos_embeddings_DEP = model.pos_embeddings_DEP.weight.data.clone()
                    best_word_fixed_embeddings_DEP = model.word_fixed_embeddings_DEP.weight.data.clone()
                    best_hidden2tag = model.hidden2tag.weight.data.clone()
                    best_MLP = model.MLP.weight.data.clone()
                    log("best dep params preserved")



                """
                if F1 < Last_SRL_score and F_label+F_link < Last_DEP_score:
                    for weight_i, last_weight_i in zip(model.BiLSTM_0.parameters(), Best_BiLSTM_0_data):
                        weight_i.data.copy_(last_weight_i)
                    for weight_i, last_weight_i in zip(model.BiLSTM_1.parameters(), Last_BiLSTM_1_data):
                        weight_i.data.copy_(last_weight_i)
                    for weight_i, last_weight_i in zip(model.BiLSTM_2.parameters(), Last_BiLSTM_2_data):
                        weight_i.data.copy_(last_weight_i)
                    log('backward!')
                """



                Last_SRL_score = F1
                Last_DEP_score = F_label + F_link





       ##########################################################################################


        tac = time.time()

        passed = tac - tic
        log("epoch %i took %f min (~%f sec per sample)" % (
            e, passed / 60, passed / sample_count
        ))





