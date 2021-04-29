import pandas as pd
import tensorflow as tf
import numpy as np
import random
import time
import os


class Generator(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, vocab_size):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.rnn = tf.keras.layers.LSTM(rnn_dim, return_state=True, return_sequences=True,
                                                kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal')
        self.vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.h_layer = tf.keras.layers.Dense(rnn_dim)

    @staticmethod
    def reconstruction_loss(x, predictions):
        temp_mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions) * temp_mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res

    def call(self, x, z):
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x.shape[0], 1)), x[:, :-1]], axis=-1)
        dec_embeddings = self.embeddings(y)
        mask = self.embeddings.compute_mask(y)
        h = self.h_layer(z)
        c = tf.zeros(shape=h.shape)
        output, _, _ = self.rnn(dec_embeddings, mask=mask, initial_state=[h, c])
        predictions = self.vocab_prob(output)
        rec_loss = self.reconstruction_loss(x, predictions)
        return predictions, rec_loss

    def greedy_decoding(self, z, maxlen):
        # 1 is the index of <bos>
        y = tf.constant(1, shape=(z.shape[0], 1))
        h = self.h_layer(z)
        c = tf.zeros(shape=h.shape)
        state = [h, c]

        res = tf.constant(0, shape=(z.shape[0], 0), dtype=tf.int64)
        for _ in range(0, maxlen):
            dec_embeddings = self.embeddings(y)
            output, h, c = self.rnn(dec_embeddings, initial_state=state)
            state = [h, c]
            pred = self.vocab_prob(output)
            y = tf.keras.backend.argmax(pred, axis=-1)
            res = tf.keras.backend.concatenate([res, y], axis=-1)
        return res


if __name__ == "__main__":
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)
    batch_size = 64
    epochs = 10
    lr = 0.01

    print("loading data")
    word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    index = 3
    with open(os.path.join(os.getcwd(), 'Dataset\\toy\\vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            word2index[vocab] = index
            index2word[index] = vocab
            index = index + 1

    # training data
    sentences = []
    maxlen = 0
    with open(os.path.join(os.getcwd(), 'Dataset\\toy\\train.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    train_df = pd.read_csv(os.path.join(os.getcwd(), 'Dataset\\toy\\train_representation1.csv'), index_col='index')
    z_train = train_df.to_numpy()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, z_train))
    train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)

    # validation data
    sentences = []
    maxlen = 0
    with open(os.path.join(os.getcwd(), 'Dataset\\toy\\valid.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_val = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    val_df = pd.read_csv(os.path.join(os.getcwd(), 'Dataset\\toy\\valid_representation1.csv'), index_col='index')
    z_val = val_df.to_numpy()
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, z_val))
    val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size)

    model = Generator(4, 16, len(word2index))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x, z):
        with tf.GradientTape() as tape:
            rec_loss = model(x, z)[-1]

            loss = tf.keras.backend.mean(rec_loss)

            grads = tape.gradient(loss, model.weights)
            optimizer.apply_gradients(zip(grads, model.weights))
            return loss

    @tf.function
    def test_step(x, z):
        rec_loss = model(x, z)[-1]
        loss = tf.keras.backend.mean(rec_loss)
        return loss

    total_loss = 0
    for step, (x_batch_val, z_batch_val) in enumerate(val_dataset):
        loss = test_step(x_batch_val, z_batch_val)
        total_loss = total_loss + loss
    val_loss = total_loss / (step + 1)
    print("loss:{:.4f}".format(val_loss))

    step_count = 1

    for epoch in range(1, epochs + 1):
        print("Start of epoch {:d}".format(epoch))
        start_time = time.time()

        total_loss = 0
        for step, (x_batch_train, z_batch_train) in enumerate(train_dataset):
            loss = train_step(x_batch_train, z_batch_train)

            total_loss = total_loss + loss

            if step_count % 100 == 0:
                print("step:{:d} train_loss:{:.4f} ".format(step_count, loss))

            step_count = step_count + 1

        train_loss = total_loss / (step + 1)

        total_loss = 0
        for step, (x_batch_val, z_batch_val) in enumerate(val_dataset):
            loss = test_step(x_batch_val, z_batch_val)
            total_loss = total_loss + loss
        val_loss = total_loss / (step + 1)
        print("loss:{:.4f}".format(val_loss))

        print("time taken:{:.2f}s".format(time.time() - start_time))

    print("model test")
    # test data
    test_df = pd.read_csv(os.path.join(os.getcwd(), 'Dataset\\toy\\test_representation1.csv'), index_col='index')
    z_test = test_df.to_numpy().tolist()

    index = random.sample(range(len(z_test)), 2)
    print(index)
    z1 = tf.constant(z_test[index[0]], shape=(1, len(z_test[index[0]])))
    z2 = tf.constant(z_test[index[1]], shape=(1, len(z_test[index[1]])))

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

    for dim in range(0, z1.shape[1]):
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