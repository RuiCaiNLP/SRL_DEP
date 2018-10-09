import argparse
import itertools
import json
import copy
import sys
import random

from util import *


conll2009_label_set = ['C-AM-EXT', 'AM-CAU', 'C-A1', 'AM-PRD',
                           'R-AA', 'C-A3', 'AM-EXT', 'R-A0',
                           'C-AM-TMP', 'C-AM-LOC', 'AM-DIR', 'R-AM-TMP', 'C-AM-NEG', 'R-A2',
                           'C-AM-DIR', 'AM-PRT', 'C-A0', 'AM-NEG', 'R-AM-PNC', 'R-AM-ADV',
                           'C-AM-MNR', 'A0', 'AM', 'AM-MNR', 'A4', 'R-AM-MNR',
                           'R-AM-EXT', 'AM-TM', 'AM-ADV', 'AA', 'AM-MOD',
                           'C-AM-ADV', 'R-AM-LOC', 'C-AM-PNC', 'A5', 'AM-TMP',
                           'C-A4', 'C-AM-CAU', 'AM-REC', 'A2', 'R-A1', 'R-AM-DIR',
                           'C-A2', 'AM-DIS', 'A1', 'C-AM-DIS', 'AM-PNC', 'C-R-AM-TMP',
                           'R-A3', 'R-A4', 'A3', 'R-AM-CAU', 'AM-LOC']


la = dict()
#file = open("conll2009.train", 'w')

def make_bio_sample(data, frames):
    frames = json.load(open(frames, 'r'))
    data = json.load(open(data, 'r'))
    data = {int(d): data[d] for d in data}
    for doc_id, sent_id, frame_name, frame_instance in frame_data(data):
        dbg_header = '%s %s %s' % (doc_id, sent_id, frame_name)

        if frame_name not in frames:
            frames[frame_name] = {
                'FEs': {
                }
            }

        frame = frames[frame_name]
        sent = data[doc_id][sent_id]['tokenized_sentence']

        role_cats = ['Core', 'Core-Unexpressed', 'Extra-Thematic',
                     'Peripheral', 'Modifiers']
        roles_cats = [cat for cat in role_cats if cat in frame['FEs']]

        roles = []
        for cat in roles_cats:
            roles += frame['FEs'][cat]
        roles = [role[0] for role in roles]

        if 'target' not in frame_instance or not \
                frame_instance['target']['index'][0]:  # probably bad record
            sys.stderr.write(
                '%s: skipping, because of null target\n' % dbg_header)
            continue

        target = frame_instance['target']['index'][0]
        target = target[-1]
        labels = frame_instance['roles']

        for label in labels:
            if label != 'O' and label not in roles:
                # sys.stderr.write("%s: cannot find %s in %s\n" % (
                #     dbg_header, label, roles))
                roles = copy.deepcopy(conll2009_label_set)


        if len(labels) != len(sent):
            raise Exception("%s: labels and sent sizes differ")
        roles_voc = roles
        roles_voc.append('O')


        if any([' ' in label for label in roles_voc]):
            sys.stderr.write(
                "%s: bad symbols in role name %s\n" % (dbg_header, roles_voc))
            continue

        def normalize(token):
            penn_tokens = {
                 '-LRB-': '(',
                 '-RRB-': ')',
                 '-LSB-': '[',
                 '-RSB-': ']',
                 '-LCB-': '{',
                 '-RCB-': '}'
            }

            if token in penn_tokens:
                return penn_tokens[token]

            token = token.lower()
            try:
                int(token)
                return "<NUM>"
            except:
                pass
            try:
                float(token.replace(',',''))
                return "<FLOAT>"
            except:
                pass
            return token

        tmp_out_ = {}
        dep_parse = []
        root_dep_parse = []
        degree = 0
        predicate_idx = target + 1
        dep_dict = {}
        dep_pairs = []
        specific_dep_pairs = []
        specific_dep_parse = []

        ROOT_node = -1

        # record with dep_parse modifier->head , dep_dict head->modifier
        LeftLabel = ['N']
        RightLabel = ['N']
        for item in data[doc_id][sent_id]['d_parsing']:
            label, tail, head = item

            LeftLabel.append('N')
            RightLabel.append('N')
            tail, head = tail[0], head[0]
            dep_parse.append("%s|%s|%s" % (label, head, tail))
            root_dep_parse.append("%s|%s|%s" % (label, head, tail))


            ## try to locate the local dependency
            dep_pairs.append([label, int(head), int(tail)])
            if dep_dict.has_key(tail):
                dep_dict[int(tail)].append(int(head))
            else:
                dep_dict[int(tail)] = [int(head)]




        child2parent = dep_pairs
        parent2child = dep_dict
        predicate = int(target) + 1
        Left_cur = predicate
        Right_cur = predicate


        #Moving to Left
        while Left_cur > 1:
            new_left_cur = Left_cur
            left_label = 'NULL'
            right_label = 'NULL'
            if child2parent[Left_cur-1][2] < new_left_cur and child2parent[Left_cur-1][2] > 0:
                new_left_cur = child2parent[Left_cur - 1][2]
                if child2parent[Left_cur - 1][0] == 'ROOT':
                    print(new_left_cur)
                left_label = child2parent[Left_cur - 1][0] + '_M'
                right_label = child2parent[Left_cur - 1][0] + '_H'

            if parent2child.has_key(Left_cur):
                for child in parent2child[Left_cur]:
                    if child < new_left_cur:
                        new_left_cur = child
                        left_label = child2parent[child - 1][0] + '_H'
                        right_label = child2parent[child - 1][0] + '_M'

            if new_left_cur > Left_cur:
                Left_cur -= 1
            else:
                LeftLabel[Left_cur] = left_label
                RightLabel[new_left_cur] = right_label
                Left_cur = new_left_cur

        #Moving to Right
        while Right_cur < len(dep_parse)-1:
            new_right_cur = Right_cur - 1
            left_label = 'NULL'
            right_label = 'NULL'
            if child2parent[Right_cur - 1][2] > new_right_cur + 1 and child2parent[Right_cur-1][2] > 0:
                new_right_cur = child2parent[Right_cur - 1][2]
                left_label = child2parent[Right_cur - 1][0] + '_H'
                right_label = child2parent[Right_cur - 1][0] + '_M'

            if parent2child.has_key(Right_cur):
                for child in parent2child[Right_cur]:
                    if child > new_right_cur and child <len(dep_parse)-1:
                        new_right_cur = child
                        left_label = child2parent[child - 1][0] + '_M'
                        right_label = child2parent[child - 1][0] + '_H'

            if new_right_cur < Right_cur:
                Right_cur += 1
            else:
                LeftLabel[new_right_cur] = left_label
                RightLabel[Right_cur] = right_label
                Right_cur = new_right_cur


        """
        up_step = 5
        down_step = 6

        sub_root = predicate


        for i in range(up_step):
            if int(child2parent[int(sub_root) - 1][2]) == 0:
                break
            sub_root = child2parent[int(sub_root) - 1][2]

        ## make predicate the new sub-root
        # child2parent[int(predicate) -1] = 0

        current_node = predicate
        old_father =  child2parent[current_node - 1][2]
        old_deptag = -1
        while current_node != sub_root:
            old_grandpa = child2parent[old_father - 1][2]
            #old_father = child2parent[current_node - 1][2]
            parent2child[old_father].remove(current_node)
            child2parent[old_father - 1][2] = current_node
            if old_deptag == -1:
                old_deptag = dep_parse[old_father - 1]
                label, head, tail = dep_parse[current_node - 1].split('|')
                dep_parse[old_father - 1] = 'rev' + label + '|' + str(old_father) + '|' + str(current_node)
            else:
                last_old_deptag = dep_parse[old_father - 1]
                dep_parse[old_father - 1] = old_deptag
                old_deptag = last_old_deptag
                label, head, tail = dep_parse[old_father - 1].split('|')
                dep_parse[old_father - 1] = 'rev' + label + '|' + str(old_father) + '|' + str(current_node)
            if parent2child.has_key(current_node):
                parent2child[current_node].append(old_father)
            else:
                parent2child[current_node] = [old_father]
            current_node = old_father
            old_father = old_grandpa

        child2parent[int(predicate) - 1][2] = 0
        label, head, tail = dep_parse[int(predicate) - 1].split('|')
        dep_parse[int(predicate) - 1] = 'Pre' + label + '|' + str(predicate) + '|' + '0'
        sub_root = predicate


        nodes_each_level = [[sub_root]]
        current_level = [sub_root]
        for i in range(down_step):
            next_level = []
            for node in current_level:
                if parent2child.has_key(node):
                    next_level.extend(parent2child[node])
            current_level = next_level
            nodes_each_level.append(current_level)


        """
        # # travel from the predicate node
        # # the distance starts from 1
        # old_tails = [predicate_idx]
        # specific_dep_pairs[predicate_idx-1][1] = 0
        # distance = 1
        # while old_tails!= []:
        #     new_tails = []
        #     for tail in old_tails:
        #         if dep_dict.has_key(tail):
        #             for head in dep_dict[tail]:
        #                 specific_dep_pairs[head - 1][1] = distance
        #                 new_tails.append(head)
        #     distance += 1
        #     old_tails = new_tails
        #
        # # travel from the root node
        # # the distance starts from 1
        # old_tails = [ROOT_node]
        # distance = 1
        # while old_tails != []:
        #     new_tails = []
        #     for tail in old_tails:
        #         if dep_dict.has_key(tail):
        #             for head in dep_dict[tail]:
        #                 specific_dep_pairs[head - 1][0] = dep_pairs[head - 1][0]
        #                 specific_dep_pairs[head - 1][2] = distance
        #                 new_tails.append(head)
        #     distance += 1
        #     old_tails = new_tails

        # for item in specific_dep_pairs:
        #     label, head, tail = item
        #     specific_dep_parse.append("%s|%s|%s" % (label, head, tail))

        #for item in nodes_each_level:
        #    trans = []
         #   for node in item:
        #        trans.append("%s:%s" % (child2parent[node-1][2], node))
        #    specific_dep_parse.append('|'.join(trans))

        sent = ' '.join([normalize(w) for w in sent])
        labels = ' '.join(labels)
        roles_voc = ' '.join(roles_voc)
        pos_tags = ' '.join(data[doc_id][sent_id]['pos'])
        dep_parse = ' '.join(dep_parse)
        root_dep_parse = ' '.join(root_dep_parse)

        left_labels = ' '.join(LeftLabel)
        right_labels = ' '.join(RightLabel)
        #specific_dep_parse = ' '.join(specific_dep_parse)


        all_targets = []
        all_lemmas = []
        for a in data[doc_id][str(doc_id)]:
            if a.startswith('f_'):
                for fr in data[doc_id][str(doc_id)][a]:
                    all_lemmas.append(a[2:])
                    all_targets.append(str(fr['target']['index'][0][0]))
        all_l = ' '.join(all_lemmas)
        all_t = ' '.join(all_targets)

        print("#%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
            dbg_header, sent, pos_tags, dep_parse, root_dep_parse, frame_name, target, all_l, all_t, roles_voc, labels, left_labels, right_labels))



        #file.write("#%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
        #    dbg_header, sent, pos_tags, dep_parse, degree, frame_name, target, all_l, all_t, roles_voc, labels))
        #file.write('\n')



def arg_parse():
    parser = argparse.ArgumentParser("SRL argument extractor")

    parser.add_argument("--data", help="json data file", required=True)

    parser.add_argument(
        "--frames", help="path to frame decriptions",
        required=True)

    return parser.parse_args()


def main():
    a = arg_parse()
    make_bio_sample(a.data, a.frames)
    #data = 'CoNLL2009-ST-English-trial.txt.jason'
    #frames = "nombank_descriptions-1.0+prop3.1.json"

if __name__ == '__main__':
    main()
