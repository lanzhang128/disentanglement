import os
import random
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


class GenerativeDataset:
    def __init__(self):
        # generative factors
        self.generative_factors = []

        # respective value range of each generative factors
        self.value_space = []

        # sentence indexes of sentences having each value
        self.sample_space = []

        # representations of sentences based on sample space
        self.representation_space = []

    def get_representation_space(self, representations):
        for i in range(0, len(self.sample_space)):
            self.representation_space.append([[] for _ in range(0, len(self.sample_space[i]))])
            for j in range(0, len(self.sample_space[i])):
                self.representation_space[i][j] = representations[self.sample_space[i][j], :]


class POSDataset(GenerativeDataset):
    def __init__(self, path):
        super().__init__()
        dic = {}
        with open(os.path.join(path, 'root.txt'), 'r') as f:
            for root in f.readlines():
                pos = root[:root.find(':')]
                temp = root[root.find(':') + 1:].split()
                for word in temp:
                    dic[word] = pos
                if len(temp) == 1:
                    continue
                self.generative_factors.append(pos)
                self.value_space.append([])
                self.sample_space.append([])

        print("generative factors:", self.generative_factors)
        with open(os.path.join(path, 'test.txt'), 'r') as f:
            sentences = f.readlines()

        for index in range(0, len(sentences)):
            sentence = sentences[index]
            sentence = sentence.rstrip()
            structure = sentence.split()
            for j in range(0, len(structure)):
                structure[j] = dic[structure[j]]
            for pos in self.generative_factors:
                if pos in structure:
                    temp = sentence.split()
                    temp_word = []
                    for i in range(0, len(temp)):
                        if dic[temp[i]] == pos:
                            temp_word.append(temp[i])
                    pos_index = self.generative_factors.index(pos)
                    if temp_word not in self.value_space[pos_index]:
                        self.value_space[pos_index].append(temp_word)
                        self.sample_space[pos_index].append([index])
                    else:
                        value_index = self.value_space[pos_index].index(temp_word)
                        self.sample_space[pos_index][value_index].append(index)


class YNOCDataset(GenerativeDataset):
    def __init__(self, path):
        super().__init__()
        with open(os.path.join(path, 'root.txt'), 'r') as f:
            for root in f.readlines():
                gf = root[:root.find(':')]
                temp = root[root.find(':') + 1:].split()
                if len(temp) == 1:
                    continue
                self.generative_factors.append(gf)
                self.value_space.append(temp)
                self.sample_space.append([[] for _ in range(0, len(temp))])

        print("generative factors:", self.generative_factors)
        with open(os.path.join(path, 'test.txt'), 'r') as f:
            sentences = f.readlines()

        for index in range(0, len(sentences)):
            sentence = sentences[index]
            sentence = sentence.rstrip()
            words = sentence.split()
            for word in words:
                for i in range(0, len(self.generative_factors)):
                    if word in self.value_space[i]:
                        self.sample_space[i][self.value_space[i].index(word)].append(index)


class Disentanglement:
    def __init__(self, datapath, representations):
        if os.path.basename(datapath) == 'pos':
            self.dataset = POSDataset(datapath)
        elif os.path.basename(datapath) == 'ynoc' or os.path.basename(datapath) == 'toy':
            self.dataset = YNOCDataset(datapath)
        else:
            raise ValueError
        self.representations = representations
        self.dataset.get_representation_space(representations)

    def group_sampling(self, generative_factor, value, batch_size):
        i = self.dataset.generative_factors.index(generative_factor)
        j = self.dataset.value_space[i].index(value)
        temp_space = self.dataset.representation_space[i][j]
        return temp_space[random.sample(range(0, temp_space.shape[0]), batch_size), :]

    def stratified_sampling(self, generative_factor, sample_number):
        i = self.dataset.generative_factors.index(generative_factor)
        p_value = [len(self.dataset.sample_space[i][j]) for j in range(0, len(self.dataset.sample_space[i]))]
        samples = []
        temp = sum(p_value)
        for j in range(0, len(p_value)):
            p_value[j] = p_value[j] / temp
            temp_space = self.dataset.representation_space[i][j]
            temp_sample_number = round(sample_number * p_value[j])
            temp_samples = temp_space[random.sample(range(0, temp_space.shape[0]), temp_sample_number), :]
            samples.append(temp_samples)
        return samples, np.array(p_value)

    @ staticmethod
    def entropy(p):
        temp = p.flatten()
        temp = temp[np.where(temp > 0)]
        return np.sum(- temp * np.log(temp))

    def mutual_information_estimation(self, num_bins, sample_number, normalize=False):
        z_max = np.max(self.representations, axis=0)
        z_min = np.min(self.representations, axis=0)
        h_z = []
        for k in range(0, self.representations.shape[1]):
            p_z = []
            temp_z = self.representations[:, k]
            bins = z_min[k] + np.arange(0, num_bins + 1) * (z_max[k] - z_min[k]) / num_bins
            for b in range(0, num_bins):
                if b == num_bins - 1:
                    temp = np.where((temp_z >= bins[b]) & (temp_z <= bins[b + 1]))
                else:
                    temp = np.where((temp_z >= bins[b]) & (temp_z < bins[b + 1]))
                p_z.append(temp[0].shape[0])
            p_z = np.array(p_z)
            p_z = p_z / np.sum(p_z)
            h_z.append(self.entropy(p_z))

        mutual_information = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples, p_value = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)
            if normalize:
                h_value = self.entropy(p_value)

            mi = [0 for _ in range(0, self.representations.shape[1])]
            for k in range(0, self.representations.shape[1]):
                h_z_given_value = 0
                bins = z_min[k] + np.arange(0, num_bins + 1) * (z_max[k] - z_min[k]) / num_bins
                for j in range(0, p_value.shape[0]):
                    p_z_given_value = []
                    temp_z = samples[j][:, k]
                    for b in range(0, num_bins):
                        if b == num_bins - 1:
                            temp = np.where((temp_z >= bins[b]) & (temp_z <= bins[b + 1]))
                        else:
                            temp = np.where((temp_z >= bins[b]) & (temp_z < bins[b + 1]))
                        p_z_given_value.append(temp[0].shape[0])
                    p_z_given_value = np.array(p_z_given_value)
                    p_z_given_value = p_z_given_value / np.sum(p_z_given_value)
                    h_z_given_value += p_value[j] * self.entropy(p_z_given_value)
                mi[k] = h_z[k] - h_z_given_value
                if normalize:
                    mi[k] = mi[k] / h_value

            mutual_information.append(mi)
        return np.array(mutual_information)

    def beta_vae_metric(self, batch_size=64, sample_number=1000):
        initial = True

        # sample for each pos
        for i in range(0, len(self.dataset.generative_factors)):
            # sample observations for classification
            index = []
            for j in range(0, len(self.dataset.sample_space[i])):
                index = index + self.dataset.sample_space[i][j]

            for b in range(0, sample_number):
                index_sample = random.sample(index, 1)[0]
                for j in range(0, len(self.dataset.sample_space[i])):
                    if index_sample in self.dataset.sample_space[i][j]:
                        break
                z1 = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                z2 = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                z_diff = np.mean(np.abs(z1 - z2), axis=0)
                z_diff.resize((1, z_diff.shape[0]))
                if initial:
                    x = z_diff
                    y = i * np.ones(shape=(1,))
                    initial = False
                else:
                    x = np.concatenate([x, z_diff], axis=0)
                    y = np.concatenate([y, i * np.ones(shape=(1,))], axis=0)

        y = tf.keras.utils.to_categorical(y)

        # randomly shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        y = y[indices, :]

        # split
        x_train, x_test = x[:int(0.8 * x.shape[0]), :], x[int(0.8 * x.shape[0]):, :]
        y_train, y_test = y[:int(0.8 * y.shape[0]), :], y[int(0.8 * y.shape[0]):, :]
        print("training points: {:d}, test points: {:d}".format(x_train.shape[0], x_test.shape[0]))

        # 10 simple linear classifiers
        acc = []
        for i in range(0, 10):
            inputs = tf.keras.Input(shape=(x.shape[1],))
            outputs = tf.keras.layers.Dense(y.shape[1], activation='softmax')(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                          loss="categorical_crossentropy", metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=0)
            test_scores = model.evaluate(x_test, y_test, verbose=0)
            acc.append(test_scores[1])
        acc = np.array(acc)
        print("Beta-VAE metric score: mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        return np.mean(acc), np.std(acc)

    def factor_vae_metric(self, batch_size=64, sample_number=1000):
        scale = np.std(self.representations, axis=0)
        initial = True

        # sample for each pos
        for i in range(0, len(self.dataset.generative_factors)):
            index = []
            for j in range(0, len(self.dataset.sample_space[i])):
                index = index + self.dataset.sample_space[i][j]

            for b in range(0, sample_number):
                index_sample = random.sample(index, 1)[0]
                for j in range(0, len(self.dataset.sample_space[i])):
                    if index_sample in self.dataset.sample_space[i][j]:
                        break
                z = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                z_var = np.var(z / scale, axis=0)
                if initial:
                    x = np.argmin(z_var) * np.ones(shape=(1,))
                    y = i * np.ones(shape=(1,))
                    initial = False
                else:
                    x = np.concatenate([x, np.argmin(z_var) * np.ones(shape=(1,))], axis=0)
                    y = np.concatenate([y, i * np.ones(shape=(1,))], axis=0)

        # 10 majority vote classifiers
        acc = []
        for i in range(0, 10):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            x_train, x_test = x[:int(0.8 * x.shape[0])], x[int(0.8 * x.shape[0]):]
            y_train, y_test = y[:int(0.8 * y.shape[0])], y[int(0.8 * y.shape[0]):]
            V = np.zeros(shape=(self.representations.shape[1], len(self.dataset.generative_factors)))
            for j in range(0, x_train.shape[0]):
                V[int(x_train[j]), int(y_train[j])] += 1
            temp = 0
            for j in range(0, x_test.shape[0]):
                if np.argmax(V[int(x_test[j]), :]) == y_test[j]:
                    temp += 1
            acc.append(temp / x_test.shape[0])
        acc = np.array(acc)
        print("Factor-VAE metric score: mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        return np.mean(acc), np.std(acc)

    def mutual_information_gap(self, num_bins=20, sample_number=10000):
        mi = self.mutual_information_estimation(num_bins, sample_number, normalize=True)
        mig = []
        for i in range(0, mi.shape[0]):
            temp_mi = mi[i, :].tolist()
            temp_mi.sort(reverse=True)
            mig.append(temp_mi[0] - temp_mi[1])
        print("Mutual Information Gap: {:.4f}".format(sum(mig) / len(mig)))
        return sum(mig) / len(mig)

    def modularity_explicitness(self, num_bins=20, sample_number=10000):
        mi = self.mutual_information_estimation(num_bins, sample_number)
        mask = np.zeros(shape=mi.shape)
        index = np.argmax(mi, axis=0)
        for i in range(0, index.shape[0]):
            mask[index[i], i] = 1
        temp_t = mi * mask
        delta = np.sum(np.square(mi - temp_t), axis=0) / (np.sum(np.square(temp_t), axis=0) * (mi.shape[0] - 1))
        modularity = 1 - delta
        print("Modularity: {:.4f}".format(np.mean(modularity)))

        explicitness = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)[0]
            for j in range(0, len(samples)):
                temp = samples[j]
                temp_train, temp_test = temp[:int(0.8 * temp.shape[0]), :], temp[int(0.8 * temp.shape[0]):, :]
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = j * np.ones(temp_train.shape[0]), j * np.ones(temp_test.shape[0])
                else:
                    x_train = np.concatenate([x_train, temp_train], axis=0)
                    x_test = np.concatenate([x_test, temp_test], axis=0)
                    y_train = np.concatenate([y_train, j * np.ones(temp_train.shape[0])], axis=0)
                    y_test = np.concatenate([y_test, j * np.ones(temp_test.shape[0])], axis=0)
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            # suggested in code from original paper
            model = LogisticRegression(C=1e10, solver='liblinear')
            model.fit(x_train, y_train)
            preds = model.predict_proba(x_test)
            roc_auc = []
            for j in range(0, len(samples)):
                y_true = (y_test == j)
                y_pred = preds[:, j]
                roc_auc.append(roc_auc_score(y_true, y_pred))
            roc_auc = np.array(roc_auc)
            explicitness.append(np.mean(roc_auc))

        explicitness = np.array(explicitness)
        print("Explicitness: {:.4f}".format(np.mean(explicitness)))
        return np.mean(modularity), np.mean(explicitness)

    def disentanglement_completeness_informativeness(self, sample_number=10000):
        informativeness = []
        r = []
        disentanglement = []
        completeness = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)[0]

            for j in range(0, len(samples)):
                temp = samples[j]
                temp_train, temp_test = temp[:int(0.8 * temp.shape[0]), :], temp[int(0.8 * temp.shape[0]):, :]
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = j * np.ones(temp_train.shape[0]), j * np.ones(temp_test.shape[0])
                else:
                    x_train = np.concatenate([x_train, temp_train], axis=0)
                    x_test = np.concatenate([x_test, temp_test], axis=0)
                    y_train = np.concatenate([y_train, j * np.ones(temp_train.shape[0])], axis=0)
                    y_test = np.concatenate([y_test, j * np.ones(temp_test.shape[0])], axis=0)
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            model = RandomForestClassifier(n_estimators=10)
            model.fit(x_train, y_train)
            informativeness.append(model.score(x_test, y_test))
            r.append(model.feature_importances_)

        r = np.array(r)
        for i in range(0, r.shape[1]):
            p = r[:, i]
            p = p / np.sum(p)
            h_k_p = self.entropy(p) / np.log(r.shape[0])
            disentanglement.append(1 - h_k_p)

        disentanglement = np.array(disentanglement)
        weight = np.sum(r, axis=0) / np.sum(r)
        print("Disentanglement Score: {:.4f}".format(np.sum(weight * disentanglement)))

        for j in range(0, r.shape[0]):
            p = r[j, :]
            p = p / np.sum(p)
            h_d_p = self.entropy(p) / np.log(r.shape[1])
            completeness.append(1 - h_d_p)

        completeness = np.array(completeness)
        print("Completeness Score: {:.4f}".format(np.mean(completeness)))

        informativeness = np.array(informativeness)
        print("Informativeness Score: {:.4f}".format(np.mean(informativeness)))

        return np.sum(weight * disentanglement), np.mean(completeness), np.mean(informativeness)

    def separated_attribute_predictability(self, sample_number=10000):
        sap = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)[0]

            for j in range(0, len(samples)):
                temp = samples[j]
                temp_train, temp_test = temp[:int(0.8 * temp.shape[0]), :], temp[int(0.8 * temp.shape[0]):, :]
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = j * np.ones(temp_train.shape[0]), j * np.ones(temp_test.shape[0])
                else:
                    x_train = np.concatenate([x_train, temp_train], axis=0)
                    x_test = np.concatenate([x_test, temp_test], axis=0)
                    y_train = np.concatenate([y_train, j * np.ones(temp_train.shape[0])], axis=0)
                    y_test = np.concatenate([y_test, j * np.ones(temp_test.shape[0])], axis=0)
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            acc = []
            for j in range(0, x_train.shape[1]):
                temp_x_train, temp_x_test = x_train[:, j].reshape(-1, 1), x_test[:, j].reshape(-1, 1)
                model = LinearSVC(C=0.01)
                model.fit(temp_x_train, y_train)
                acc.append(model.score(temp_x_test, y_test))
            acc.sort(reverse=True)
            sap.append(acc[0] - acc[1])
        print("SAP score: {:.4f}".format(np.mean(sap)))
        return np.mean(sap)


def main(mode, datapath, csv_path, seed=0):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csv_path)
    representations = np.array(df)[:, 1:]
    print('{:d} representation, {:d} dimension'.format(representations.shape[0], representations.shape[1]))

    dis = Disentanglement(datapath, representations)

    if mode == 0:
        score = dis.beta_vae_metric()[0]
    elif mode == 1:
        score = dis.factor_vae_metric()[0]
    elif mode == 2:
        score = dis.mutual_information_gap()
    elif mode == 3:
        score = dis.modularity_explicitness()[0]
    elif mode == 4:
        score = dis.disentanglement_completeness_informativeness()[0]
    elif mode == 5:
        score = dis.separated_attribute_predictability()
    else:
        score = None

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='disentanglement metrics')
    tm_help = 'test mode: ' \
              '0 will use Beta-VAE metric (Higgins et al., 2016), ' \
              '1 will use FactorVAE metric (Kim & Mnih, 2018), ' \
              '2 will use Mutual Information Gap metric (Chen et al., 2018), ' \
              '3 will use modularity and explicitness metric (Ridgeway & Mozer, 2018), ' \
              '4 will use disentanglement, completeness and informativeness metric (Eastwood & Williams, 2018), ' \
              '5 will use separated attribute predictability metric (Kumar et al., 2018).'
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help=tm_help)
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-d', '--data', default='toy', help='dataset')
    parser.add_argument('-f', '--file', default='Dataset\\toy\\test_representation1.csv', help='csv file path')

    args = parser.parse_args()

    mode = args.test_mode
    seed = args.seed
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), args.data)
    csv_file = args.file

    if mode not in range(0, 6):
        print("wrong mode, please type disentanglement.py -h for help")
        raise ValueError
    print("Dataset:", args.data)
    main(mode, datapath, csv_file, seed)
