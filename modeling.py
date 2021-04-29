import tensorflow as tf
import numpy as np
import time
import os


class LSTMAE(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, z_dim, vocab_size):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.encoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True,
                                                kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal')
        self.decoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=True, return_sequences=True,
                                                kernel_initializer='lecun_normal',recurrent_initializer='lecun_normal')
        self.decoder_vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.encoder_z_layer = tf.keras.layers.Dense(z_dim)
        self.decoder_h_layer = tf.keras.layers.Dense(rnn_dim)

    def encoding(self, x):
        enc_embeddings = self.embeddings(x)
        mask = self.embeddings.compute_mask(x)
        output = self.encoder_rnn(enc_embeddings, mask=mask)
        temp_mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 2))
        temp_mask = tf.keras.backend.expand_dims(temp_mask)
        temp_mask = tf.keras.backend.repeat_elements(temp_mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * temp_mask, axis=1)
        z = self.encoder_z_layer(output)
        return z

    def decoder_training(self, x, z):
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x.shape[0], 1)), x[:, :-1]], axis=-1)
        dec_embeddings = self.embeddings(y)
        mask = self.embeddings.compute_mask(y)
        h = self.decoder_h_layer(z)
        c = tf.zeros(shape=h.shape)
        output, _, _ = self.decoder_rnn(dec_embeddings, mask=mask, initial_state=[h, c])
        predictions = self.decoder_vocab_prob(output)
        return predictions

    @staticmethod
    def reconstruction_loss(x, predictions):
        temp_mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions) * temp_mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res

    def call(self, x):
        z = self.encoding(x)
        predictions = self.decoder_training(x, z)

        rec_loss = self.reconstruction_loss(x, predictions)
        return z, predictions, rec_loss

    def train(self, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset):
        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                rec_loss = self(x)[-1]

                loss = tf.keras.backend.mean(rec_loss)

            grads = tape.gradient(loss, self.weights)
            optimizer.apply_gradients(zip(grads, self.weights))
            return loss

        @tf.function
        def test_step(x):
            rec_loss = self(x)[-1]
            loss = tf.keras.backend.mean(rec_loss)
            return loss

        total_loss = 0
        for step, x_batch_val in enumerate(val_dataset):
            loss = test_step(x_batch_val)
            total_loss = total_loss + loss
        val_loss = total_loss / (step + 1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("loss:{:.4f}\n".format(val_loss))
        print("loss:{:.4f}\n".format(val_loss))

        if epochs <= 0:
            ckpt_man.save()

        step_count = 1

        for epoch in range(1, epochs + 1):
            print("Start of epoch {:d}".format(epoch))
            start_time = time.time()

            total_loss = 0
            for step, x_batch_train in enumerate(train_dataset):
                loss = train_step(x_batch_train)

                total_loss = total_loss + loss

                if step_count % 100 == 0:
                    print("step:{:d} train_loss:{:.4f} ".format(step_count, loss))

                step_count = step_count + 1

            train_loss = total_loss / (step + 1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("train_loss:{:.4f} ".format(train_loss))
            print("train_loss:{:.4f}".format(train_loss))

            total_loss = 0
            for step, x_batch_val in enumerate(val_dataset):
                loss = test_step(x_batch_val)

                total_loss = total_loss + loss

            val_loss = total_loss / (step + 1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("loss:{:.4f}\n".format(val_loss))
            print("loss:{:.4f}".format(val_loss))

            ckpt_man.save()
            print("time taken:{:.2f}s".format(time.time() - start_time))

        print('training ends, model at {}'.format(ckpt_dir))

    def test(self, test_dataset):
        @tf.function
        def test_step(x):
            rec_loss = self(x)[-1]
            loss = tf.keras.backend.mean(rec_loss)
            return loss

        print("model test")
        total_loss = 0
        for step, x_batch_test in enumerate(test_dataset):
            loss = test_step(x_batch_test)
            total_loss = total_loss + loss

            if step == 0:
                all_mean = self.encoding(x_batch_test)
            else:
                mean = self.encoding(x_batch_test)
                all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)
        test_loss = total_loss / (step + 1)
        print("loss:{:.4f}\n".format(test_loss))
        all_mean = all_mean.numpy()
        cov = np.cov(all_mean, rowvar=False)
        s = []
        n = []
        for i in range(0, cov.shape[0]):
            if cov[i][i] > 0.01:
                s.append(i + 1)
            else:
                n.append(i + 1)
        print('{} active units:{}'.format(len(s), s))
        print('{} inactive units:{}'.format(len(n), n))
        return test_loss, len(s)

    def get_representation(self, test_dataset):
        print("get mean vector (representation) for sentences")
        initial = True
        for x_batch_test in test_dataset:
            if initial:
                all_mean = self.encoding(x_batch_test)
                initial = False
            else:
                mean = self.encoding(x_batch_test)
                all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)
        all_mean = all_mean.numpy()
        return all_mean

    def greedy_decoding(self, z, maxlen):
        # 1 is the index of <bos>
        y = tf.constant(1, shape=(z.shape[0], 1))
        h = self.decoder_h_layer(z)
        c = tf.zeros(shape=h.shape)
        state = [h, c]

        res = tf.constant(0, shape=(z.shape[0], 0), dtype=tf.int64)
        for _ in range(0, maxlen):
            dec_embeddings = self.embeddings(y)
            output, h, c = self.decoder_rnn(dec_embeddings, initial_state=state)
            state = [h, c]
            pred = self.decoder_vocab_prob(output)
            y = tf.keras.backend.argmax(pred, axis=-1)
            res = tf.keras.backend.concatenate([res, y], axis=-1)
        return res


class LSTMVAE(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, z_dim, vocab_size, z_mode=0, bi=False):
        super().__init__()
        self.z_mode = z_mode
        if self.z_mode not in range(0, 4):
            print('Wrong mode for the latent code.')
            raise ValueError
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        if bi:
            self.encoder_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_state=False,
                                                                                  return_sequences=True,
                                                                                  kernel_initializer='lecun_normal',
                                                                                  recurrent_initializer='lecun_normal'))
        else:
            self.encoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True,
                                                    kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.encoder_mean_layer = tf.keras.layers.Dense(z_dim)
        self.encoder_logvar_layer = tf.keras.layers.Dense(z_dim)
        self.decoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=True, return_sequences=True,
                                                kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.decoder_vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')
        if self.z_mode == 0 or self.z_mode == 2:
            self.decoder_h_layer = tf.keras.layers.Dense(rnn_dim)

    def encoding(self, x):
        enc_embeddings = self.embeddings(x)
        mask = self.embeddings.compute_mask(x)
        output = self.encoder_rnn(enc_embeddings, mask=mask)

        # extract the whole sentence representation, 2 is the index of <eos>
        temp_mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 2))
        temp_mask = tf.keras.backend.expand_dims(temp_mask)
        temp_mask = tf.keras.backend.repeat_elements(temp_mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output*temp_mask, axis=1)

        # get latent code with diagonal Gaussian
        mean = self.encoder_mean_layer(output)
        logvar = self.encoder_logvar_layer(output)
        epsilon = tf.random.normal(shape=mean.shape)
        z = mean + tf.keras.backend.exp(0.5 * logvar) * epsilon

        return z, mean, logvar

    def decoder_training(self, x, z):
        # 1 is the index of <bos>
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x.shape[0], 1)), x[:, :-1]], axis=-1)
        dec_embeddings = self.embeddings(y)
        mask = self.embeddings.compute_mask(y)
        if self.z_mode == 0:
            h = self.decoder_h_layer(z)
            c = tf.zeros(shape=h.shape)
            output, _, _ = self.decoder_rnn(dec_embeddings, mask=mask, initial_state=[h, c])
        elif self.z_mode == 1:
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            output, _, _ = self.decoder_rnn(dec_input, mask=mask)
        elif self.z_mode == 2:
            h = self.decoder_h_layer(z)
            c = tf.zeros(shape=h.shape)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            output, _, _ = self.decoder_rnn(dec_input, mask=mask, initial_state=[h, c])
        else:
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            output, _, _ = self.decoder_rnn(new_z, mask=mask)
        predictions = self.decoder_vocab_prob(output)
        return predictions

    @staticmethod
    def kld_loss(mean, logvar):
        kld = 0.5 * tf.keras.backend.sum(tf.keras.backend.square(mean) + tf.keras.backend.exp(logvar) - 1 - logvar,
                                         axis=-1)
        return kld

    @staticmethod
    def reconstruction_loss(x, predictions):
        # ignore padding
        temp_mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions)*temp_mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res

    def call(self, x):
        z, mean, logvar = self.encoding(x)

        # calculate KL divergence
        kl_loss = self.kld_loss(mean, logvar)

        predictions = self.decoder_training(x, z)

        rec_loss = self.reconstruction_loss(x, predictions)
        elbo = kl_loss + rec_loss
        return predictions, kl_loss, rec_loss, elbo, mean, logvar

    def train(self, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, beta, C_target, C_step=5000):
        @tf.function
        def train_step(x, C_value):
            with tf.GradientTape() as tape:
                kl_loss, rec_loss, elbo = self(x)[1:4]

                loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_value) * beta + rec_loss)

            grads = tape.gradient(loss, self.weights)
            optimizer.apply_gradients(zip(grads, self.weights))

            elbo = tf.keras.backend.mean(elbo)
            kl_loss = tf.keras.backend.mean(kl_loss)
            rec_loss = tf.keras.backend.mean(rec_loss)
            return kl_loss, rec_loss, elbo, loss

        @tf.function
        def test_step(x):
            kl_loss, rec_loss, elbo = self(x)[1:4]

            loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_target) * beta + rec_loss)
            elbo = tf.keras.backend.mean(elbo)
            kl_loss = tf.keras.backend.mean(kl_loss)
            rec_loss = tf.keras.backend.mean(rec_loss)
            return kl_loss, rec_loss, elbo, loss

        total_loss = 0
        total_kl_loss = 0
        total_rec_loss = 0
        total_elbo = 0
        for step, x_batch_val in enumerate(val_dataset):
            kl_loss, rec_loss, elbo, loss = test_step(x_batch_val)

            total_loss = total_loss + loss
            total_elbo = total_elbo + elbo
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

        val_loss, val_kl_loss, val_rec_loss, val_elbo = \
            total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_elbo / (
                        step + 1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}\n".format(
                val_loss, val_kl_loss, val_rec_loss, val_elbo))
        print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(
            val_loss, val_kl_loss, val_rec_loss, val_elbo))

        if epochs <= 0:
            ckpt_man.save()

        # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
        step_count = 1

        print('loss=beta({:f})*|kl-C({:f})|+rec)'.format(beta, C_target))

        for epoch in range(1, epochs+1):
            print("Start of epoch {:d}".format(epoch))
            start_time = time.time()

            total_loss = 0
            total_kl_loss = 0
            total_rec_loss = 0
            total_elbo = 0
            for step, x_batch_train in enumerate(train_dataset):
                if step_count <= C_step and C_target > 0:
                    C_value = C_target * step_count / C_step
                else:
                    C_value = C_target
                kl_loss, rec_loss, elbo, loss = train_step(x_batch_train, tf.constant(C_value*1.0))

                total_loss = total_loss + loss
                total_elbo = total_elbo + elbo
                total_rec_loss = total_rec_loss + rec_loss
                total_kl_loss = total_kl_loss + kl_loss

                if step_count % 100 == 0:
                    print("step:{:d} train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f}"
                          .format(step_count, loss, kl_loss, rec_loss, elbo))

                step_count = step_count + 1

            train_loss, train_kl_loss, train_rec_loss, train_elbo = \
                total_loss/(step+1), total_kl_loss/(step+1), total_rec_loss/(step+1), total_elbo/(step+1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f} ".format(
                    train_loss, train_kl_loss, train_rec_loss, train_elbo))
            print("train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f}".format(
                train_loss, train_kl_loss, train_rec_loss, train_elbo))

            total_loss = 0
            total_kl_loss = 0
            total_rec_loss = 0
            total_elbo = 0
            for step, x_batch_val in enumerate(val_dataset):
                kl_loss, rec_loss, elbo, loss = test_step(x_batch_val)

                total_loss = total_loss + loss
                total_elbo = total_elbo + elbo
                total_rec_loss = total_rec_loss + rec_loss
                total_kl_loss = total_kl_loss + kl_loss

            val_loss, val_kl_loss, val_rec_loss, val_elbo = \
                total_loss/(step+1), total_kl_loss/(step+1), total_rec_loss/(step+1), total_elbo/(step+1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}\n".format(
                    val_loss, val_kl_loss, val_rec_loss, val_elbo))
            print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(
                val_loss, val_kl_loss, val_rec_loss, val_elbo))

            ckpt_man.save()
            print("time taken:{:.2f}s".format(time.time() - start_time))

        print('training ends, model at {}'.format(ckpt_dir))

    def test(self, test_dataset):
        @tf.function
        def test_step(x):
            kl_loss, rec_loss, elbo = self(x)[1:4]

            elbo = tf.keras.backend.mean(elbo)
            kl_loss = tf.keras.backend.mean(kl_loss)
            rec_loss = tf.keras.backend.mean(rec_loss)
            return kl_loss, rec_loss, elbo

        print("model test")
        total_kl_loss = 0
        total_rec_loss = 0
        total_elbo = 0
        for step, x_batch_test in enumerate(test_dataset):
            kl_loss, rec_loss, elbo = test_step(x_batch_test)

            total_elbo = total_elbo + elbo
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

            if step == 0:
                all_mean = self.encoding(x_batch_test)[-2]
            else:
                mean = self.encoding(x_batch_test)[-2]
                all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)

        test_kl_loss, test_rec_loss, test_elbo = \
            total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_elbo / (step + 1)
        print("kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(test_kl_loss, test_rec_loss, test_elbo))

        all_mean = all_mean.numpy()
        cov = np.cov(all_mean, rowvar=False)
        s = []
        n = []
        for i in range(0, cov.shape[0]):
            if cov[i][i] > 0.01:
                s.append(i + 1)
            else:
                n.append(i + 1)
        print('{} active units:{}'.format(len(s), s))
        print('{} inactive units:{}'.format(len(n), n))
        return test_kl_loss, test_rec_loss, test_elbo, len(s)

    def get_representation(self, test_dataset):
        print("get mean vector (representation) for sentences")
        initial = True
        for x_batch_test in test_dataset:
            if initial:
                all_mean = self.encoding(x_batch_test)[-2]
                initial = False
            else:
                mean = self.encoding(x_batch_test)[-2]
                all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)
        all_mean = all_mean.numpy()
        return all_mean

    def greedy_decoding(self, z, maxlen):
        # 1 is the index of <bos>
        y = tf.constant(1, shape=(z.shape[0], 1))
        if self.z_mode == 0 or self.z_mode == 2:
            h = self.decoder_h_layer(z)
            c = tf.zeros(shape=h.shape)
            state = [h, c]
        else:
            state = None

        res = tf.constant(0, shape=(z.shape[0], 0), dtype=tf.int64)
        for _ in range(0, maxlen):
            dec_embeddings = self.embeddings(y)
            if self.z_mode == 0:
                output, h, c = self.decoder_rnn(dec_embeddings, initial_state=state)
            elif self.z_mode == 1 or self.z_mode == 2:
                new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
                dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
                output, h, c = self.decoder_rnn(dec_input, initial_state=state)
            else:
                new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
                output, h, c = self.decoder_rnn(new_z, initial_state=state)
            state = [h, c]
            pred = self.decoder_vocab_prob(output)
            y = tf.keras.backend.argmax(pred, axis=-1)
            res = tf.keras.backend.concatenate([res, y], axis=-1)
        return res