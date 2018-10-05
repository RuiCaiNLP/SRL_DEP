file_in = open("conll2009_batch.test.dep_l", "r")

file_out = open("conll2009_decompose.dev", "w")

pattern_sentence = dict()

for line in file_in.readlines():
    parts = line.strip().split("\t")
    predicate = parts[0].split()[-1]
    predicate_idx = int(parts[-7])
    sentence = parts[1].split()
    labels = parts[-3].split()
    print(sentence[predicate_idx], predicate_idx, sentence, labels)

    pattern = []
    marked_sentence = []
    idx = 0
    for word, label in zip(sentence, labels):
        if word == ".":
            if len(pattern) > 0 and pattern[-1] == "CONTEXT":
                marked_sentence.append("_)")
            continue
        if idx == predicate_idx:
            label = "P"
        idx += 1
        if label != "O":
            if len(pattern) > 0 and pattern[-1] == "CONTEXT":
                marked_sentence.append("_)")
            pattern.append(label)
            marked_sentence.append(word)
        elif len(pattern) == 0 or pattern[-1] != "CONTEXT":
            pattern.append("CONTEXT")
            marked_sentence.append("(_")
            marked_sentence.append(word)
        else:
            marked_sentence.append(word)

    pattern = "\t".join(pattern)
    marked_sentence = "\t".join(marked_sentence)
    if pattern_sentence.has_key(pattern):
        pattern_sentence[pattern].append(marked_sentence)
    else:
        pattern_sentence[pattern] = [marked_sentence]


for key, value in pattern_sentence.items():
    file_out.write(key)
    file_out.write("\n")
    for sentence in value:
        file_out.write(sentence)
        file_out.write("\n")
    file_out.write("\n")
