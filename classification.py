import os
import random
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification task')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='DBPedia_C_15_0', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'classification.txt')):
        df = pd.DataFrame(columns=['Model', 'Mean', 'Std'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'classification.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'classification.txt'))

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dic = {'Model': os.path.basename(model_path)}
        with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
            s = f.readlines()[0]
            s = s.split(',')

        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
        x_df = pd.read_csv(os.path.join(model_path, 'representation.csv'), index_col='index')
        x = np.array(x_df)

        labels = []
        with open(os.path.join(datapath, 'test_class_label.txt'), 'r') as f:
            for label in f.readlines():
                label = int(label.rstrip())
                labels.append(label)
        y = np.array(labels)
        y = tf.keras.utils.to_categorical(y)

        x, y = shuffle(x, y, random_state=seed)
        x_train, x_test = x[:int(0.8 * x.shape[0]), :], x[int(0.8 * x.shape[0]):, :]
        y_train, y_test = y[:int(0.8 * y.shape[0]), :], y[int(0.8 * y.shape[0]):, :]
        print("training points: {:d}, test points: {:d}".format(x_train.shape[0], x_test.shape[0]))
        acc = []
        for i in range(0, 10):
            inputs = tf.keras.Input(shape=(x.shape[1],))
            d1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
            d2 = tf.keras.layers.Dense(128, activation='relu')(d1)
            outputs = tf.keras.layers.Dense(y.shape[1], activation='softmax')(d2)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss="categorical_crossentropy", metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2, verbose=0)
            test_scores = model.evaluate(x_test, y_test, verbose=0)
            acc.append(test_scores[1])
        acc = np.array(acc)
        print("mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        dic['Mean'] = np.mean(acc)
        dic['Std'] = np.std(acc)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'classification.txt'), index=False, float_format='%.6f')