import os
import random
import argparse
import tensorflow as tf
import numpy as np
from modeling import LSTMAE, LSTMVAE
from nltk.translate.bleu_score import corpus_bleu


def load_dic(datapath):
    word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    index = 3
    with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            word2index[vocab] = index
            index2word[index] = vocab
            index = index + 1
    return word2index, index2word


def load_model(mt, z_mode, emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path):
    if mt == 'AE':
        model = LSTMAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size)
    elif mt == 'VAE':
        model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size, z_mode=z_mode)
    else:
        model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size, z_mode=z_mode, bi=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    return model


def mean_vector_reconstruct(model, mt, datapath, reconstruction_file, word2index, index2word, batch_size):
    print("mean vector reconstruction")
    maxlen = 0
    sentences = []
    with open(os.path.join(datapath, 'test.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)
    f = open(reconstruction_file, 'w')
    for x_batch_test in test_dataset:
        if mt == 'AE':
            mean = model.encoding(x_batch_test)
        else:
            mean = model.encoding(x_batch_test)[0]
        z = mean
        input = tf.constant(word2index['<bos>'], shape=(x_batch_test.shape[0], 1), dtype=tf.int64)
        state = None
        output = input
        for _ in range(maxlen):
            dec_embeddings = model.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = model.decoder_rnn(dec_input, initial_state=state)
            pred = model.decoder_vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output = tf.keras.backend.concatenate([output, pred], axis=1)

        output = output[:, 1:]
        output = output.numpy().tolist()
        for element in output:
            if word2index['<eos>'] in element:
                element = element[:element.index(word2index['<eos>'])]
            element = [index2word[i] for i in element]
            f.write(' '.join(element) + '\n')
    f.close()
    print("reconstruction file at :{}.".format(reconstruction_file))


def bleu(candidate_path, reference_path):
    references = []
    with open(reference_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip().split()
            references.append([sentence])

    candidates = []
    with open(candidate_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip().split()
            candidates.append(sentence)

    bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1:{:f}, BLEU-2:{:f}, BLEU-4:{:f}'.format(bleu1*100, bleu2*100, bleu4*100))


def homotopy(mt, model, z_dim, datapath, index2word):
    maxlen = 0
    sentences = []
    with open(os.path.join(datapath, 'test.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)
    index = random.sample(range(len(sentences)), 2)
    print(index)

    if mt == 'AE':
        z1 = model.encoding(tf.constant(sentences[index[0]], shape=(1, len(sentences[index[0]]))))
        z2 = model.encoding(tf.constant(sentences[index[1]], shape=(1, len(sentences[index[1]]))))
    else:
        z1 = model.encoding(tf.constant(sentences[0], shape=(1, len(sentences[0]))))[1]
        z2 = model.encoding(tf.constant(sentences[1], shape=(1, len(sentences[1]))))[1]
    print(z1)
    print(z2)

    print('normal homotopy:')
    for i in range(0, 6):
        print(i + 1, end='. ')
        z = (1 - 0.2 * i) * z1 + 0.2 * i * z2
        res = model.greedy_decoding(z, maxlen).numpy().tolist()[0]
        for j in range(0, len(res)):
            if index2word[int(res[j])] == '<eos>':
                break
            else:
                print(index2word[int(res[j])], end=' ')
        print()

    for dim in range(0, z_dim):
        print('dim {:d} homotopy'.format(dim + 1))
        print('from {:.3f} to {:.3f}'.format(z1.numpy()[0, dim], z2.numpy()[0, dim]))
        for i in range(0, 6):
            print(i + 1, end='. ')
            z = z1.numpy()
            z[0, dim] = (1 - 0.2 * i) * z1.numpy()[0, dim] + 0.2 * i * z2.numpy()[0, dim]
            z = tf.constant(z)
            res = model.greedy_decoding(z, maxlen).numpy().tolist()[0]
            for j in range(0, len(res)):
                if index2word[int(res[j])] == '<eos>':
                    break
                else:
                    print(index2word[int(res[j])], end=' ')
            print()
        z1 = z


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    tm_help = 'test mode: ' \
              '0 will reconstruct test set and report BLEU scores for reconstruction file, ' \
              '1 will do homotopy evaluation.'
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help=tm_help)
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='toy_capacity', help='path of model')

    args = parser.parse_args()

    mode = args.test_mode
    seed = args.seed
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)
    print(model_path)

    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[0]
        s = s.split(',')

    mt = s[0].split()[-1]
    if mt == 'AE':
        z_mode = 0
        emb_dim = int(s[1].split()[-1])
        rnn_dim = int(s[2].split()[-1])
        z_dim = int(s[3].split()[-1])
        batch_size = int(s[4].split()[-1])
        lr = float(s[6].split()[-1])
    else:
        z_mode = int(s[1].split()[-1])
        emb_dim = int(s[2].split()[-1])
        rnn_dim = int(s[3].split()[-1])
        z_dim = int(s[4].split()[-1])
        batch_size = int(s[5].split()[-1])
        lr = float(s[7].split()[-1])
    datapath = s[-2].split()[-1]
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    model = load_model(mt, z_mode, emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    random.seed(seed)

    if mode == 0:
        mean_file = os.path.join(model_path, 'mean.txt')
        mean_vector_reconstruct(model, mt, datapath, mean_file, word2index, index2word, batch_size)
        reference_path = os.path.join(datapath, 'test.txt')
        print('mean reconstruction file bleu scores (reference original file)')
        bleu(mean_file, reference_path)
    elif mode == 1:
        homotopy(mt, model, z_dim, datapath, index2word)
    else:
        print("wrong mode, please type quality.py -h for help")
