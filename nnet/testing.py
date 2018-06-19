import torch
from nnet.util import *
import torch.autograd
from torch.autograd import Variable
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NullTester(object):
    def __init__(self):
        self.best = float('inf')

    def compute_error(self, *args, **kwargs):
        return 0,0,0

    def run(self, *args, **kwargs):
        pass



def make_local_voc(labels):
    return {i: label for i, label in enumerate(labels)}

def syntax_analyse(tree, predicate_index):
    child2parent = []
    parent2child = {}
    root_node = -1
    for node in tree:
        child2parent.append(list(node))
        if node[2] == 0:
            root_node = node[1]
        if parent2child.has_key(node[2]):
            parent2child[node[2]].append(node[1])
        else:
            parent2child[node[2]] = [node[1]]
    assert root_node > 0




    #make predicate the new root
    current_node = predicate_index + 1
    old_father = child2parent[current_node-1][2]
    while current_node != root_node:
        old_grandpa = child2parent[old_father-1][2]
        parent2child[old_father].remove(current_node)
        child2parent[old_father - 1][2] = current_node
        if parent2child.has_key(current_node):
            parent2child[current_node].append(old_father)
        else:
            parent2child[current_node] = [old_father]

        current_node = old_father
        old_father = old_grandpa

    #travel from predicate
    covered = []
    distances = [-1]*len(tree)
    current_dis = 0
    current_level = [predicate_index+1]
    while len(covered) < len(tree):
        covered.extend(current_level)
        next_level = []
        for node in current_level:
            distances[node-1] = current_dis
            if parent2child.has_key(node):
                next_level.extend(parent2child[node])
        current_dis = current_dis + 1
        current_level = next_level

    return distances






def test(model, train_set, test_set, converter, params_path):
    log("start just test...")
    losses, errors, errors_w, NonNullPredicts, right_NonNullPredicts, NonNullTruths = 0., 0, 0., 0., 0., 0.
    log('now fucking batch computing')
    index = 0
    model.to(device)
    NonNullTruth_dis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Right_predict_dis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    NonNullPre_dis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    P_dis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    R_dis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    F_dis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    model.eval()
    best_F1 = -0.1
    result_file = open('model_bilstm_4L_seed1_30_256_only_deptags_gold_dev.result', 'w')
    for gold_batch, batch in zip(train_set.batches(), test_set.batches()):
        #log(batch[0])
        #log(batch[0][1][4])
        index += 1
        log(index)
        # loss, e, e_w, NonNullPredict, right_NonNullPredict, NonNullTruth = self.error_computer.compute(model, batch)
        errors, errors_w = 0, 0.0
        NonNullPredict = 0
        NonNullTruth = 0
        right_NonNullPredict = 0

        record_ids, batch = zip(*batch)

        record_ids, gold_batch = zip(*gold_batch)

        sentences = []
        syntax_distances = []
        for record in gold_batch:
            sentences.append(record[1])
            syntax_distances.append(syntax_analyse(record[4], record[6]))

        # model.test_mode_on()

        model_input = converter(batch)
        model.hidden = model.init_hidden_share()
        model.hidden_2 = model.init_hidden_spe()
        model.hidden_3 = model.init_hidden_spe()
        model.hidden_4 = model.init_hidden_spe()

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

        SRLloss, DEPloss, SPEDEPloss, loss, SRLprobs, wrong_l_nums, all_l_nums, spe_wrong_l_nums, spe_all_l_nums \
            = model(sentence_in, p_sentence_in, pos_tags_in, sen_lengths, target_idx_in, region_mark_in,
                    local_roles_voc_in,
                    frames_in, local_roles_mask_in, sent_pred_lemmas_idx_in, dep_tags_in, dep_heads,
                    targets, specific_dep_tags_in, specific_dep_relations_in)

        labels = np.argmax(SRLprobs.cpu().data.numpy(), axis=1)
        labels = np.reshape(labels, sentence.shape)



        iof_values = []


        for i, sent_labels in enumerate(labels):
            labels_voc = batch[i][-4]
            local_voc = make_local_voc(labels_voc)

            """

            format = '%10s\t' * len(sentences[i]) + '\n'
            result_file.write(format % tuple(sentences[i]))
            format = '%10d\t' * len(sentences[i]) + '\n'
            result_file.write(format % tuple(syntax_distances[i]))

            for level in range(len(iof_values)):

                i_value_forward = []
                i_value_backward = []
                f_value_forward = []
                f_value_backward = []
                o_value_forward = []
                o_value_backward = []

                iof_values_forward= iof_values[level][0].data.numpy()
                iof_values_backward = iof_values[level][1].data.numpy()

                for timestep in range(len(sentences[i])):
                    i_value_forward.append(float(iof_values_forward[i][timestep][0]))
                    f_value_forward.append(float(iof_values_forward[i][timestep][1]))
                    o_value_forward.append(float(iof_values_forward[i][timestep][2]))

                    i_value_backward.append(float(iof_values_backward[i][timestep][0]))
                    f_value_backward.append(float(iof_values_backward[i][timestep][1]))
                    o_value_backward.append(float(iof_values_backward[i][timestep][2]))

                result_file.write('(the i, f, o gates value of level %d )\n' % level)
                format = '%10.2f\t' * len(sentences[i]) + '\n'
                result_file.write(format % tuple(i_value_forward))
                result_file.write(format % tuple(i_value_backward))
                result_file.write('\n')

                format = '%10.2f\t' * len(sentences[i]) + '\n'
                result_file.write(format % tuple(f_value_forward))
                result_file.write(format % tuple(f_value_backward))
                result_file.write('\n')

                format = '%10.2f\t' * len(sentences[i]) + '\n'
                result_file.write(format % tuple(o_value_forward))
                result_file.write(format % tuple(o_value_backward))


        """

            best_labels = []
            true_labels = []
            for j in range(len(labels[i])):
                best = local_voc[labels[i][j]]
                true = local_voc[tags[i][j]]
                if true != '<pad>':
                    best_labels.append(best)
                    true_labels.append(true)
                if true != '<pad>' and true != 'O':
                    NonNullTruth += 1
                    #NonNullTruth_dis[int(syntax_distances[i][j])] += 1
                if true != best:
                    errors += 1
                if best != '<pad>' and best != 'O' and true != '<pad>':
                    NonNullPredict += 1
                    #NonNullPre_dis[int(syntax_distances[i][j])] += 1
                    if true == best:
                        right_NonNullPredict += 1
                        #Right_predict_dis[int(syntax_distances[i][j])] += 1
            #format = '%10s\t' * len(sentences[i]) + '\n'
            #result_file.write(format % tuple(best_labels))
            #format = '%10s\t' * len(sentences[i]) + '\n'
            #result_file.write(format % tuple(true_labels))
            #result_file.write('\n')
        NonNullPredicts += NonNullPredict
        right_NonNullPredicts += right_NonNullPredict
        NonNullTruths += NonNullTruth

    for i in range(20):
        break
        if NonNullTruth_dis[i] == 0:
            continue

        P_dis[i] = Right_predict_dis[i]*1.0 / (NonNullPre_dis[i] + 0.0001)
        R_dis[i] = Right_predict_dis[i]*1.0 / (NonNullTruth_dis[i] + 0.0001)
        F_dis[i] = 2 * P_dis[i] * R_dis[i]/(P_dis[i] + R_dis[i] + 0.0001)

    Predicat_num = 6390
    P = (right_NonNullPredicts + Predicat_num*0.9438) / (NonNullPredicts + Predicat_num)
    R = (right_NonNullPredicts + Predicat_num*0.9438) / (NonNullTruths + Predicat_num)
    F1 = 2 * P * R / (P + R)
    log(right_NonNullPredicts)
    log(NonNullPredicts)
    log(NonNullTruths)
    log('Precision: ' + str(P), 'recall: ' + str(R), 'F1: ' + str(F1))
    P = (right_NonNullPredicts) / (NonNullPredicts + 0.000001)
    R = (right_NonNullPredicts) / (NonNullTruths)
    F1 = 2 * P * R / (P + R + 0.0001)
    log('Precision: ' + str(P), 'recall: ' + str(R), 'F1: ' + str(F1))
    log('Best F1: ' + str(best_F1))

