from collections import Counter
from basemodel import Hps
from basemodel import BaseModel
import tensorflow as tf
import numpy as np
import random
from create_mini_train_corpus import func
from basemodel import HPS


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


models = [None, None, None]
sessions = [None, None, None]

file_paths = [line.strip() for line in open('model_paths.txt')]

parameters = [
    (HPS[0], file_paths[0]),
    (HPS[1], file_paths[1]),
    (HPS[2], file_paths[2]),
]


def get_model_i_result(x, model_index):
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


def get_three_predictions(x):
    results = zip(get_model_i_result(x, 0),
                  get_model_i_result(x, 1),
                  get_model_i_result(x, 2))

    results = list(results)

    majorities = [get_marjority(r) for r in results]

    return majorities


X = []
Y = []

for i in range(10000):
    span = 2000000
    r_x = random.randrange(-span, span)
    r_y = random.randrange(-span, span)

    X.append([r_x, r_y])
    Y.append(func(r_x, r_y))

label = np.array(Y)

index = 0
result = get_model_i_result(X, index)
result = np.array(result)
precision = np.sum(result == label) / len(label)
print('model {} precision is {}'.format(index + 1, precision))
index = 1
result = get_model_i_result(X, index)
result = np.array(result)
precision = np.sum(result == label) / len(label)
print('model {} precision is {}'.format(index + 1, precision))
index = 2
result = get_model_i_result(X, index)
result = np.array(result)
precision = np.sum(result == label) / len(label)
print('model {} precision is {}'.format(index + 1, precision))

predicated = get_three_predictions(X)
print('conflict number: {}'.format(conflict_num))
predicated = np.array(predicated)
precision = np.sum(predicated == label) / len(label)
print('final precision is {}'.format(precision))


# with tf.Graph().as_default():
#     model, session = load_model
#     output = session.run(model.eval(x))
#     predicate_label = np.argmax(output, axis=1)
#     print(predicate_label)
#
# with tf.Graph().as_default():
#     model, session = load_model
#     output = session.run(model.eval(x))
#     predicate_label = np.argmax(output, axis=1)
#     print(predicate_label)
#
#