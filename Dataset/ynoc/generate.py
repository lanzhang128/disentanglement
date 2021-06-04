import os
import random


def generate_and_split(text_dir, seed=0):
    year = [str(i) for i in range(2011, 2021)]
    name = ['Name'+str(i) for i in range(1, 41)]
    occupation = ['Occupation'+str(i) for i in range(1, 21)]
    city = ['City'+str(i) for i in range(1, 31)]

    with open(os.path.join(text_dir, 'root.txt'), 'w') as f:
        f.write('year: ' + ' '.join(year) + '\n')
        f.write('name: ' + ' '.join(name) + '\n')
        f.write('occupation: ' + ' '.join(occupation) + '\n')
        f.write('city: ' + ' '.join(city) + '\n')

    random.seed(seed)
    temp_sentences = [[], [], []]
    for y in year:
        for n in name:
            for o in occupation:
                for c in city:
                    temp_sentences[0].append(' '.join(['in', y, ',', n, 'was', 'a/an', o, 'in', c, '.\n']))
                    temp_sentences[1].append(' '.join(['in', y, '\'s', c, ',', n, 'was', 'a/an', o, '.\n']))
                    temp_sentences[2].append(' '.join([n, 'was', 'a/an', o, 'in', c, 'in', y, '.\n']))
    sentences = temp_sentences[0] + temp_sentences[1] + temp_sentences[2]

    random.shuffle(sentences)
    random.shuffle(sentences)
    random.shuffle(sentences)
    with open(os.path.join(text_dir, 'train.txt'), 'w') as f:
        f.writelines(sentences[:int(0.6 * len(sentences))])
    with open(os.path.join(text_dir, 'valid.txt'), 'w') as f:
        f.writelines(sentences[int(0.6 * len(sentences)):int(0.8 * len(sentences))])
    with open(os.path.join(text_dir, 'test.txt'), 'w') as f:
        f.writelines(sentences[int(0.8 * len(sentences)):])

    voc = year + name + occupation + city + ['in', ',', 'was', 'a/an', '.', '\'s']
    random.shuffle(voc)
    random.shuffle(voc)
    with open(os.path.join(text_dir, 'vocab.txt'), 'w') as f:
        f.writelines('\n'.join(voc))


if __name__ == '__main__':
    generate_and_split(os.getcwd())