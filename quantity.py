import os
import random
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import modeling
import disentanglement


def load_model(mt, z_mode, emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path):
    if mt == 'AE':
        model = modeling.LSTMAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size)
    elif mt == 'VAE':
        model = modeling.LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size, z_mode=z_mode)
    else:
        model = modeling.LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size, z_mode=z_mode, bi=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='ynoc\\ynoc_AE_0', help='path of model')

    args = parser.parse_args()
    seed = args.seed
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'quantity.txt')):
        df = pd.DataFrame(columns=['Model', 'KL', 'Rec.', 'AU', 'Beta-VAE', 'FactorVAE', 'Modularity', 'MIG', 'DCI', 'SAP'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'quantity.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'quantity.txt'))

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
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
        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
        vocab_size = int(s[-1].split()[-1])
        model = load_model(mt, z_mode, emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

        dic = {'Model': os.path.basename(model_path)}
        word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        index = 3
        with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
            for vocab in f.readlines():
                vocab = vocab.rstrip()
                word2index[vocab] = index
                index = index + 1

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

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if mt == 'AE':
            rec, au = model.test(test_dataset)
        else:
            kl, rec, elbo, au = model.test(test_dataset)
            dic['KL'] = float(kl)
        dic['Rec.'] = float(rec)
        dic['AU'] = au
        if dic['Model'].split('_')[0] == 'pos' or dic['Model'].split('_')[0] == 'toy' or dic['Model'].split('_')[0] == 'ynoc':
            csv_file = os.path.join(model_path, 'representation.csv')
            dic['Beta-VAE'] = disentanglement.main(0, datapath, csv_file, seed)
            dic['FactorVAE'] = disentanglement.main(1, datapath, csv_file, seed)
            dic['MIG'] = disentanglement.main(2, datapath, csv_file, seed)
            dic['Modularity'] = disentanglement.main(3, datapath, csv_file, seed)
            dic['DCI'] = disentanglement.main(4, datapath, csv_file, seed)
            dic['SAP'] = disentanglement.main(5, datapath, csv_file, seed)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'quantity.txt'), index=False, float_format='%.6f')