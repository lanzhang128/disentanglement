import os
import random


def generate_and_split(text_dir, seed=0):
    A = ['A'+str(i) for i in range(1, 21)]
    B = ['B'+str(i) for i in range(1, 21)]
    C = ['C'+str(i) for i in range(1, 21)]
    D = ['D'+str(i) for i in range(1, 21)]

    with open(os.path.join(text_dir, 'root.txt'), 'w') as f:
        f.write('A: ' + ' '.join(A) + '\n')
        f.write('B: ' + ' '.join(B) + '\n')
        f.write('C: ' + ' '.join(C) + '\n')
        f.write('D: ' + ' '.join(D) + '\n')

    random.seed(seed)
    sentences = []
    for a in A:
        for b in B:
            for c in C:
                for d in D:
                    sentences.append(' '.join([a, b, c, d+'\n']))

    random.shuffle(sentences)
    random.shuffle(sentences)
    random.shuffle(sentences)
    with open(os.path.join(text_dir, 'train.txt'), 'w') as f:
        f.writelines(sentences[:int(0.6 * len(sentences))])
    with open(os.path.join(text_dir, 'valid.txt'), 'w') as f:
        f.writelines(sentences[int(0.6 * len(sentences)):int(0.8 * len(sentences))])
    with open(os.path.join(text_dir, 'test.txt'), 'w') as f:
        f.writelines(sentences[int(0.8 * len(sentences)):])

    voc = A + B + C + D
    random.shuffle(voc)
    random.shuffle(voc)
    with open(os.path.join(text_dir, 'vocab.txt'), 'w') as f:
        f.writelines('\n'.join(voc))


if __name__ == '__main__':
    generate_and_split(os.getcwd())