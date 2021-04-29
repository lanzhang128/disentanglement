import tensorflow as tf
import numpy as np
import random
import argparse
import os
import modeling
import pandas as pd


def load_data(batch_size, path):
    print("loading data")
    word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    index = 3
    with open(os.path.join(path, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            word2index[vocab] = index
            index2word[index] = vocab
            index = index + 1

    # training data
    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'train.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)

    # validation data
    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'valid.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_val = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size)

    # test data
    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'test.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)

    print("number of training data:{:d}, number of validation data:{:d}, number of test data:{:d}"
          .format(len(x_train), len(x_val), len(x_test)))
    return train_dataset, val_dataset, test_dataset, word2index, index2word


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script', epilog='start training')
    parser.add_argument('-e', '--emb_dim', default=16, type=int, help='embedding dimensions, default: 16')
    parser.add_argument('-r', '--rnn_dim', default=64, type=int, help='RNN dimensions, default: 64')
    parser.add_argument('-z', '--z_dim', default=4, type=int, help='latent space dimensions, default: 4')
    parser.add_argument('-b', '--batch', default=256, type=int, help='batch size, default: 256')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate, default: 0.001')
    parser.add_argument('--epochs', default=10, type=int, help='epochs number, default: 10')
    parser.add_argument('--datapath', default='ynoc', help='path of data under dataset directory, default: ynoc')
    parser.add_argument('-mt', '--model_type', default='VAE', help='model type, default: VAE')
    parser.add_argument('-zm', '--z_mode', type=int, default=0, help='z mode, default: 0')
    parser.add_argument('-beta', default=1, type=float, help='beta for training VAE, default: 1')
    parser.add_argument('-C', default=0, type=float, help='C for training VAE, default: 0')
    parser.add_argument('-s', '--seed', default=0, type=int, help='global random seed')
    parser.add_argument('-m', '--mpath', default='test_3', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch
    lr = args.learning_rate
    epochs = args.epochs
    mt = args.model_type
    z_mode = args.z_mode
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), args.datapath)
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    # https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset, val_dataset, test_dataset, word2index, index2word = load_data(batch_size, datapath)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()
    beta = args.beta
    C = args.C
    emb_dim = args.emb_dim
    rnn_dim = args.rnn_dim
    z_dim = args.z_dim
    if mt == 'AE':
        model = modeling.LSTMAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index))
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write(
                "training configure: model type {}, embedding dimension {:d}, RNN dimension {:d}, z dimension {:d}, "
                "batch size {:d}, epoch number {:d}, learning rate {:f}, dataset {}, vocabulary size {:d}\n"
                .format(mt, emb_dim, rnn_dim, z_dim, batch_size, epochs, lr, datapath, len(word2index)))

        model.train(optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset)

        model.test(test_dataset)
        representations = model.get_representation(test_dataset)
        mean_df = pd.DataFrame(representations)
        mean_df.columns = ['dim' + str(i) for i in range(1, z_dim + 1)]
        mean_df.to_csv(os.path.join(ckpt_dir, 'representation.csv'), index_label='index')

    elif mt == 'VAE' or mt == 'BiVAE':
        if mt == 'VAE':
            model = modeling.LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index), z_mode=z_mode)
        else:
            model = modeling.LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index), z_mode=z_mode, bi=True)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write(
                "training configure: model type {}, z mode {:d}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(mt, z_mode, emb_dim, rnn_dim, z_dim, batch_size, epochs, lr, beta, C, datapath, len(word2index)))

        model.train(optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, beta, C)

        model.test(test_dataset)
        representations = model.get_representation(test_dataset)
        mean_df = pd.DataFrame(representations)
        mean_df.columns = ['dim' + str(i) for i in range(1, z_dim + 1)]
        mean_df.to_csv(os.path.join(ckpt_dir, 'representation.csv'), index_label='index')

    else:
        print('model type does not exist!')
        exit()