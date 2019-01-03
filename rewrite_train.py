file = open('conll2009.train.dep_l', 'r')
file2 = open('conll2009_batch.train.dep_l', 'w')
idx = 1
train_lines = []
import random
for line in file.readlines():
    #file2.write(line)
    train_lines.append(line)
    if idx == 178800:
        break
    idx += 1
print(idx)
#random.shuffle(train_lines)

for line in train_lines:
    file2.write(line)

file.close()
file2.close()


file = open('conll2009.dev.dep_l', 'r')
file2 = open('conll2009_batch.dev.dep_l', 'w')
idx = 1

for line in file.readlines():
    file2.write(line)
    if idx == 6360:
        break
    idx += 1
print(idx)
file.close()
file2.close()
