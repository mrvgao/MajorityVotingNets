from collections import Counter
from basemodel import BaseModel
import tensorflow as tf
from hyperparamters import HPS
import numpy as np
import random
from functools import lru_cache
from load_cifar_10 import unpickle
import pickle


conflict_num = 0


def get_marjority(elements):
    elements = list(map(int, elements))
    if len(set(elements)) != 1:
        global conflict_num
        conflict_num += 1
    return list(Counter(elements).most_common(1)[0])[0]


def load_model(hps, model_path):
    session = tf.Session()
    model = BaseModel(hps)

    saver = tf.train.Saver()
    saver.restore(session, save_path=model_path)
    # x ** 2 < 50

    return model, session


def get_predicate_from_model(x, model, session):
    return session.run(model.eval(x))


def get_model_i_result(x, model_index):
    models = [None, None, None]
    sessions = [None, None, None]

    file_paths = [line.strip() for line in open('model_paths.txt')]

    parameters = [
        (HPS[0], file_paths[0]),
        (HPS[1], file_paths[1]),
        (HPS[2], file_paths[2]),
    ]

    tf.reset_default_graph()
    with tf.Graph().as_default():
    #     if models[model_index] is None:
        model, session = load_model(*parameters[model_index])
        models[model_index], sessions[model_index] = model, session
        # else:
        #     model, session = models[model_index], sessions[model_index]

        output = session.run(model.eval(x))
        predicate_label = np.argmax(output, axis=1)
        return predicate_label


def get_three_predictions(x, agree_number=2):
    results = zip(get_model_i_result(x, 0),
                  get_model_i_result(x, 1),
                  get_model_i_result(x, 2))

    results = list(results)

    majorities = []

    agreed_indices = []
    for ii, r in enumerate(results):
        assert len(r) == 3
        if len(set(r)) <= 3 - agree_number + 1:
            agreed_indices.append(ii)
            majorities.append(get_marjority(r))
        else:
            continue

    return majorities, agreed_indices


@lru_cache(maxsize=128)
def get_test_x_y():
    # X_test, y_test = [], []
    # with open('dataset/test.txt') as f:
    #     for ii, line in enumerate(f):
    #         if ii == 0: continue
    #
    #         values = line.strip().split()
    #         X_test.append(list(map(int, values[:-1])))
    #         y_test.append(int(values[-1]))
    #
    # return X_test, y_test

    d = unpickle(get_cifar_10_set('test'))

    X_test = d[b'data']
    y_test = d[b'labels']

    return X_test, y_test


def get_test_set_precision(model_index):
    X_test, y_test = get_test_x_y()
    result = get_model_i_result(X_test, model_index)
    result = np.array(result)
    precision = np.sum(result == np.array(y_test)) / len(y_test)
    return precision


def get_unlabel_data(file_path):
    d = unpickle(file_path)
    unlabel_X = d[b'data']

    return unlabel_X


def get_cifar_10_set(index):
    url = 'dataset/cifar-10-batches-py/'
    if index == 'test':
        url += 'test_batch'
    else:
        url += 'data_batch_{}'.format(index)

    return url


def save_labled_data(file_name, x, y):
    assert len(x) == len(y)
    print('new labled data size: {}'.format(len(x)))
    with open(file_name, 'wb') as f:
        d = {b'data': x, b'labels': y}
        pickle.dump(d, f)
    return file_name


LOOP = 0


def get_precision_of_ensemble():
    print('single test set precision: \n')
    print(get_test_set_precision(0))
    print(get_test_set_precision(1))
    print(get_test_set_precision(2))

    print('ensenmble precision is : \n')
    x, y = get_test_x_y()
    predicated, agree_indices = get_three_predictions(x, agree_number=1)
    precision = np.sum(np.array(predicated) == y) / len(y)
    print('-- {}'.format(precision))

    with open('precision_log', 'a') as f:
        f.write('\n precision: {}'.format(precision))


def merge_two_dataset(x1, x2, y1, y2):
    # labeled_ratio = 0.7  # new data : original data
    # total_length = len(y1)
    # x1_number = int(total_length * labeled_ratio)
    # x1_indices = np.arange(len(y1))
    # np.random.shuffle(x1_indices)
    # x1 = np.array(x1)[x1_indices][:x1_number]
    # y1 = np.array(y1)[x1_indices][:x1_number]
    #
    # x2_number = int(total_length * (1 - labeled_ratio))
    # x2_indices = np.arange(len(y2))
    # np.random.shuffle(x2_indices)
    # x2 = np.array(x2)[x2_indices][:x2_number]
    # y2 = np.array(y2)[x2_indices][:x2_number]

    assert len(x1) == len(y1)
    assert len(x2) == len(y2)

    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    indices = np.arange(len(y))
    np.random.shuffle(indices)

    x, y = x[indices], y[indices]

    return x, y


def read_unlabel_data(labeled_x_y, unlabel_dataset_index, loop):
    unlabel_data = get_unlabel_data(get_cifar_10_set(unlabel_dataset_index))
    predicated, agree_indices = get_three_predictions(unlabel_data, agree_number=2)
    print('conflict number: {}'.format(conflict_num))
    print('data set size is {}'.format(len(unlabel_data)))
    print('agreed number is {}'.format(len(agree_indices)))

    create_label_dataset = 'dataset/cifar10_new_label_{}'.format(loop)
    label_x, label_y = labeled_x_y
    new_label_data = unlabel_data[agree_indices]
    label_x, label_y = merge_two_dataset(label_x, new_label_data, label_y, predicated)

    save_labled_data(create_label_dataset, label_x, label_y)

    return create_label_dataset
