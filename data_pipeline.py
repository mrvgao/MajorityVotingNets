import tensorflow as tf
import random
from collections import namedtuple
from load_cifar_10 import unpickle

BatchInput = namedtuple('batch_input', ['initializer', 'x', 'y'])


def parser_tsv(line):
    line = tf.string_split([line], delimiter='\t').values
    # num = tf.string_to_number(line[1])
    num = tf.string_split([line[1]]).values
    num = tf.string_to_number(num)
    num = tf.cast(num, tf.int32)
    # num = line[:-1]
    label = tf.cast(tf.string_to_number((line[0])), tf.int32)
    num = tf.cast(num, tf.int32)
    # label = tf.cast(tf.string_to_number(line[-1]), tf.int32)
    return num, label


def one_hot_parser(numbers, labels):
    NUM_CLASS = 10
    one_hot = tf.one_hot(labels, depth=NUM_CLASS)
    return numbers, one_hot


def get_train_batch(file_name, batch_size=128):

    d = unpickle(file_name)
    X = d[b'data']
    Y = d[b'labels']

    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)

    data_X = tf.data.Dataset.from_tensor_slices(X)
    data_y = tf.data.Dataset.from_tensor_slices(Y)

    dataset = tf.data.Dataset.zip((data_X, data_y))
    dataset = dataset.map(one_hot_parser)
    # dataset = (tf.data.TextLineDataset(file_name, buffer_size=10)
    #     .skip(1)
    #     .map(parser_tsv)
    #     .map(one_hot_parser)
    # )

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    x, y = iterator.get_next()
    return BatchInput(initializer=iterator.initializer, x=x, y=y)


UnlableBatchInput = namedtuple('batch_input', ['initializer', 'x'])


def get_unlable_data(file_name, batch_size=128):
    dataset = (tf.data.TextLineDataset(file_name, buffer_size=10)
        .skip(1)
        .map(lambda line: tf.string_to_number(line))
        .map(lambda n: tf.cast(n, tf.int32))
    )

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    x = iterator.get_next()
    return UnlableBatchInput(initializer=iterator.initializer, x=x)


if __name__ == '__main__':
    with tf.Session() as sess:
        input = get_train_batch('dataset/cifar-10-batches-py/data_batch_1', batch_size=28)
        sess.run(input.initializer)
        while True:
            try:
                x, y = sess.run([input.x, input.y])
                print(x[0])
                print(y[0])
                print(x.shape)
                print(y.shape)
            except tf.errors.OutOfRangeError:
                break

        # unlabel_input = get_unlable_data('dataset/unlabel_data.txt')
        # sess.run(unlabel_input.initializer)
        #
        # x = sess.run([unlabel_input.x])
        # print(x)

