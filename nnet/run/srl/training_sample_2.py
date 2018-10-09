import argparse
import itertools
import json
import copy
import sys

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

Predicate_labels_set = []

def findpath(parent_list, current_node):
    path = [current_node]
    while current_node!=0:
        parent_node = parent_list[current_node-1][2]
        current_node = parent_node
        path.append(current_node)
    return path[::-1]


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


        dep_parse = []
        root_dep_parse = []

        child2parent_pairs = []

        predicate = target + 1
        # record with dep_parse modifier->head , dep_dict head->modifier
        Predicate_specific_Labels = []
        Predicate_specific_Relation = []
        Predicate_link = ['3' for _ in labels]
        Pair_label_table = [['0' for _ in labels] for _ in labels]
        for item in data[doc_id][sent_id]['d_parsing']:
            label, tail, head = item
            Predicate_specific_Labels.append('NULL')
            Predicate_specific_Relation.append('0')

            tail, head = tail[0], head[0]
            dep_parse.append("%s|%s|%s" % (label, head, tail))
            root_dep_parse.append("%s|%s|%s" % (label, head, tail))

            if head == predicate:
                if tail != 0:
                    Predicate_link[tail-1] = '2'
            elif tail == predicate:
                Predicate_link[head-1] ='1'


            # label, child, parent
            child2parent_pairs.append([label, int(head), int(tail)])
            if int(head)>0 and int(tail)>0:
                Pair_label_table[int(head)-1][int(tail)-1] = label
                Pair_label_table[int(tail)-1][int(head)-1] = label





        augments_set = []
        for i in range(len(labels)):
            if labels[i] != 'O':
                augments_set.append(i+1)




        # root -> predicate
        Predicate_path = findpath(child2parent_pairs, predicate)

        for augment in augments_set:
            # root->augment
            if augment == predicate:
                continue
            augment_path = findpath(child2parent_pairs, augment)
            nearest_ancestor = Predicate_path[0]


            for i in range(100):
                if i>= len(Predicate_path) or i>=len(augment_path):
                    break
                if Predicate_path[i] == augment_path[i]:
                    nearest_ancestor = Predicate_path[i]
                else:
                    break
            # old version: 1 child, 2 parent, 3 different branch
            # new version: 1 child, 2 parent, 3 different branch, 4 dencent, 5 ancestor
            #print(predicate, augment, nearest_ancestor, Predicate_path, augment_path)
            #from augument to subroot (all children), and then root to predicate(all parents)
            if nearest_ancestor!= augment and nearest_ancestor!=predicate:
                for node in augment_path[::-1]:
                    if node > 0:
                        if node == nearest_ancestor:
                            break
                        #new_label = child2parent_pairs[node-1][0]+'_R'
                        new_label = child2parent_pairs[node - 1][0]
                        Predicate_specific_Labels[node - 1] = new_label
                        Predicate_specific_Relation[node -1] = '3'
                        if new_label not in Predicate_labels_set:
                            Predicate_labels_set.append(new_label)

                start = False
                for i in range(len(Predicate_path)):
                    if Predicate_path[i] == nearest_ancestor:
                        start=True
                    if Predicate_path[i] == predicate:
                        break
                    if not start:
                        continue
                    #new_label = Pair_label_table[Predicate_path[i]-1][Predicate_path[i+1]-1]+'_H'
                    new_label = Pair_label_table[Predicate_path[i] - 1][Predicate_path[i + 1] - 1]
                    Predicate_specific_Labels[Predicate_path[i]-1] = new_label
                    Predicate_specific_Relation[Predicate_path[i] - 1] = '2'
                    if new_label not in Predicate_labels_set:
                        Predicate_labels_set.append(new_label)


            # from augment to predicate, all children
            if nearest_ancestor == predicate:

                for node in augment_path[::-1]:
                    if node == predicate:
                        break
                    if node > 0 :
                        new_label = child2parent_pairs[node-1][0]+'_C'
                        Predicate_specific_Labels[node-1] = new_label
                        Predicate_specific_Relation[node - 1] = '1'
                        if new_label not in Predicate_labels_set:
                            Predicate_labels_set.append(new_label)

            # from predicate to augment ,all parents
            if nearest_ancestor == augment:
                Predicate_path_2 = Predicate_path[::-1]
                for i in range(len(Predicate_path_2)):
                    if Predicate_path_2[i] == augment:
                        break
                    new_label = Pair_label_table[Predicate_path_2[i] - 1][Predicate_path_2[i + 1] - 1] + '_H'
                    Predicate_specific_Labels[Predicate_path_2[i + 1] - 1] = new_label
                    Predicate_specific_Relation[Predicate_path_2[i + 1] - 1] = '2'
                    if new_label not in Predicate_labels_set:
                        Predicate_labels_set.append(new_label)

        sent = ' '.join([normalize(w) for w in sent])
        labels = ' '.join(labels)
        roles_voc = ' '.join(roles_voc)
        pos_tags = ' '.join(data[doc_id][sent_id]['pos'])
        dep_parse = ' '.join(dep_parse)
        root_dep_parse = ' '.join(root_dep_parse)

        Predicate_specific_Labels = ' '.join(Predicate_specific_Labels)
        Predicate_specific_Relation = ' '.join(Predicate_specific_Relation)
        Predicate_link = ' '.join(Predicate_link)

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
            dbg_header, sent, pos_tags, dep_parse, root_dep_parse, frame_name, target, all_l, all_t, roles_voc, labels,
            Predicate_specific_Labels, Predicate_link))



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
    file = open('Specific_Dep_2.voc', 'w')
    for label in Predicate_labels_set:
        file.write(label)
        file.write('\n')
    #data = 'CoNLL2009-ST-English-trial.txt.jason'
    #frames = "nombank_descriptions-1.0+prop3.1.json"

if __name__ == '__main__':
    main()
