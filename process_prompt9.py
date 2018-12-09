# -*- coding:utf-8 -*-
import os
from reader import *

def get_all_essays(file_dir):
    all_files = os.listdir(file_dir)
    fo = open('all_essays.txt', 'w')
    count = 0
    for fi in all_files:
        path = '%s/%s' % (file_dir, fi)
        f = open(path, 'r')
        lines = f.readlines()

        if len(lines) > 4:
            score = lines[1].strip().replace('#', '')
            if len(score) <= 2 and score != '' and score != '##':
                fo.write(str(score) + ' ')
                for line in lines[3:]:
                    essay = line.strip().decode('ascii', 'ignore')
                    fo.write(essay.encode('utf8') + ' ')
                fo.write('\n')
                count += 1
    fo.close()
    print "total", count, "essays"
    return count

def write_to_tsv(count, filename):
    f = open(filename, 'r')
    lines = f.readlines()
    train_num = int(count * 0.6)
    dev_num = int(count * 0.2)

    id = 13000

    ff = open('../data/training_set_rel3.tsv', 'a+')

    with open('../data/fold_0/train.tsv', 'a+') as f1:
        for i in range(0, train_num):
            items = lines[i].strip().split()
            essay = ' '.join(items[1:])
            s = (str(id), '9', essay, items[0])
            line = '\t'.join(s)
            f1.write(line.encode('utf8') + '\n')
            ff.write(line.encode('utf8') + '\n')
            id += 1

    with open('../data/fold_0/dev.tsv', 'a+') as f2:
        for j in range(train_num, train_num + dev_num):
            items = lines[j].strip().split()
            essay = ' '.join(items[1:])
            s = (str(id), '9', essay, items[0])
            line = '\t'.join(s)
            f2.write(line.encode('utf8') + '\n')
            ff.write(line.encode('utf8') + '\n')
            id += 1

    with open('../data/fold_0/test.tsv', 'a+') as f3:
        for x in range(train_num + dev_num, count):
            items = lines[x].strip().split()
            essay = ' '.join(items[1:])
            s = (str(id), '9', essay, items[0])
            line = '\t'.join(s)
            f3.write(line.encode('utf8') + '\n')
            ff.write(line.encode('utf8') + '\n')
            id += 1
    ff.close()

def get_data_info():

    f = open('all_essays.txt', 'r')
    lines = f.readlines()
    scores = []
    total_length = 0
    count = 0
    for line in lines:
        items = line.strip().split()
        if items[0] != "##":
            score = int(items[0])
            scores.append(score)
            essay = ' '.join(items[1:])
            tokens = tokenize(essay)
            total_length += len(tokens)
            count += 1
    scores.sort()
    median = scores[len(scores) // 2]
    print 'avg_length:', int(total_length / float(count))
    print 'med:', median
    print count

if __name__ == "__main__":
    # count = get_all_essays('./data/prompt#9')
    # write_to_tsv(count, 'all_essays.txt')
    get_data_info()