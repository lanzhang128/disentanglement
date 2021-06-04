import os
import numpy as np
import copy
import pandas as pd

if __name__ == '__main__':
    np.random.seed(0)
    words = []
    with open(os.path.join(os.getcwd(), 'root.txt'), 'r') as f:
        for line in f.readlines():
            words.append(line.split(':')[1].split())
    values1 = copy.deepcopy(words)
    values2 = copy.deepcopy(words)
    for i in range(0, len(words)):
        for j in range(0, len(words[i])):
            values1[i][j] = np.random.uniform(-1, 1) * np.ones(1)
            values2[i][j] = np.random.uniform(-1, 1, size=2)

    with open(os.path.join(os.getcwd(), 'train.txt'), 'r') as f:
        sentences = f.readlines()

    representations1 = []
    representations2 = []
    for i in range(0, len(sentences)):
        temp = sentences[i].rstrip().split()
        temp_rep1 = np.zeros(0)
        temp_rep2 = np.zeros(0)

        for k in range(0, len(words)):
            for j in range(0, len(temp)):
                if temp[j] in words[k]:
                    temp_rep1 = np.concatenate([temp_rep1, np.array(values1[k][words[k].index(temp[j])])])
                    temp_rep2 = np.concatenate([temp_rep2, values2[k][words[k].index(temp[j])]])

        representations1.append(temp_rep1)
        representations2.append(temp_rep2)

    df = pd.DataFrame(representations1)
    df.columns = ['dim' + str(i) for i in range(1, len(representations1[0])+1)]
    df.to_csv(os.path.join(os.getcwd(), 'train_representation1.csv'), index_label='index')
    df = pd.DataFrame(representations2)
    df.columns = ['dim' + str(i) for i in range(1, len(representations2[0])+1)]
    df.to_csv(os.path.join(os.getcwd(), 'train_representation2.csv'), index_label='index')

    with open(os.path.join(os.getcwd(), 'valid.txt'), 'r') as f:
        sentences = f.readlines()

    representations1 = []
    representations2 = []
    for i in range(0, len(sentences)):
        temp = sentences[i].rstrip().split()
        temp_rep1 = np.zeros(0)
        temp_rep2 = np.zeros(0)

        for k in range(0, len(words)):
            for j in range(0, len(temp)):
                if temp[j] in words[k]:
                    temp_rep1 = np.concatenate([temp_rep1, np.array(values1[k][words[k].index(temp[j])])])
                    temp_rep2 = np.concatenate([temp_rep2, values2[k][words[k].index(temp[j])]])

        representations1.append(temp_rep1)
        representations2.append(temp_rep2)

    df = pd.DataFrame(representations1)
    df.columns = ['dim' + str(i) for i in range(1, len(representations1[0])+1)]
    df.to_csv(os.path.join(os.getcwd(), 'valid_representation1.csv'), index_label='index')
    df = pd.DataFrame(representations2)
    df.columns = ['dim' + str(i) for i in range(1, len(representations2[0])+1)]
    df.to_csv(os.path.join(os.getcwd(), 'valid_representation2.csv'), index_label='index')

    with open(os.path.join(os.getcwd(), 'test.txt'), 'r') as f:
        sentences = f.readlines()

    representations1 = []
    representations2 = []
    for i in range(0, len(sentences)):
        temp = sentences[i].rstrip().split()
        temp_rep1 = np.zeros(0)
        temp_rep2 = np.zeros(0)

        for k in range(0, len(words)):
            for j in range(0, len(temp)):
                if temp[j] in words[k]:
                    temp_rep1 = np.concatenate([temp_rep1, np.array(values1[k][words[k].index(temp[j])])])
                    temp_rep2 = np.concatenate([temp_rep2, values2[k][words[k].index(temp[j])]])

        representations1.append(temp_rep1)
        representations2.append(temp_rep2)

    df = pd.DataFrame(representations1)
    df.columns = ['dim' + str(i) for i in range(1, len(representations1[0])+1)]
    df.to_csv(os.path.join(os.getcwd(), 'test_representation1.csv'), index_label='index')
    df = pd.DataFrame(representations2)
    df.columns = ['dim' + str(i) for i in range(1, len(representations2[0])+1)]
    df.to_csv(os.path.join(os.getcwd(), 'test_representation2.csv'), index_label='index')