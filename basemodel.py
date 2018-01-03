import tensorflow as tf
from data_pipeline import get_train_batch
import random
import numpy as np
from hyperparamters import Hps
import os
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


VARIABLES = 'variables'

random.seed(0)


class BaseModel:
    dtype = tf.float32

    def __init__(self, hps: Hps, iterator=None, is_train=True):
        self.hps = hps
        self.__build_layer()

        if is_train and iterator:
            self.loss, self.op, self.summary, self.global_steps = \
                self.get_loss_with_x_y(iterator.x, iterator.y)

    def __build_layer(self):
        self.input_x = tf.placeholder(dtype=self.dtype, shape=[self.hps.batch_size, ])
        with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
            self.w = tf.get_variable(
                'w', [self.hps.x_size, self.hps.hidden_layers[0]],
                dtype=BaseModel.dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)
            )
        tf.add_to_collection(VARIABLES, self.w)

        with tf.variable_scope('bias', reuse=tf.AUTO_REUSE):
            self.b = tf.get_variable(
                'b', [],
                dtype=BaseModel.dtype, initializer=tf.zeros_initializer
            )

        with tf.variable_scope('w2', reuse=tf.AUTO_REUSE):
            self.w2 = tf.get_variable(
                'w2', [self.hps.hidden_layers[0], self.hps.hidden_layers[1]],
                dtype=BaseModel.dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)
            )

        tf.add_to_collection(VARIABLES, self.w2)

        with tf.variable_scope('b2', reuse=tf.AUTO_REUSE):
            self.b2 = tf.get_variable(
                'b2', [],
                dtype=BaseModel.dtype, initializer=tf.zeros_initializer
            )

        with tf.variable_scope('w3', reuse=tf.AUTO_REUSE):
            self.w3 = tf.get_variable(
                'w3', [self.hps.hidden_layers[1], self.hps.y_size],
                dtype=BaseModel.dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)
            )

        tf.add_to_collection(VARIABLES, self.w3)

        with tf.variable_scope('b3', reuse=tf.AUTO_REUSE):
            self.b3 = tf.get_variable(
                'b3', [],
                dtype=BaseModel.dtype, initializer=tf.zeros_initializer
            )

    def get_loss(self, logits, y):
        loss = tf.losses.softmax_cross_entropy(y, logits)

        tf.summary.scalar('loss', loss)

        l2_loss = tf.nn.l2_loss
        loss += self.hps.regularization * tf.reduce_sum([l2_loss(self.w),
                                                         l2_loss(self.w2),
                                                         l2_loss(self.b),
                                                         l2_loss(self.b2),
                                                         l2_loss(self.w3),
                                                         l2_loss(self.b3)])

        return loss

    def eval(self, x):
        x = tf.cast(x, self.dtype)
        output_1 = tf.matmul(x, self.w) + self.b
        output_1 = tf.nn.leaky_relu(output_1)
        # predicate = tf.nn.sigmoid(predicate)

        output_2 = tf.matmul(output_1, self.w2) + self.b2
        output_2 = tf.nn.leaky_relu(output_2)

        output_3 = tf.matmul(output_2, self.w3) + self.b3
        # output_2 = tf.sigmoid(output_2)
        return output_3

    def get_loss_with_x_y(self, x, y):
        output = self.eval(x)
        loss = self.get_loss(output, y)
        op, global_steps = self.optimize(loss)
        summary = tf.summary.merge_all()

        return loss, op, summary, global_steps

    def optimize(self, loss):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.hps.learning_rate, global_step,
                                                   1000, 0.90, staircase=False)
        op = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                .minimize(loss, global_step=global_step))
        return op, global_step


def train(hps, train_corpus, model_path=None):
    tf.reset_default_graph()

    epoch = hps.epoch
    mark = "2_dimensional_total_50_hidden_layer_{}_epoch_{}".format(hps.hidden_layers[0], epoch)

    iterator = get_train_batch(train_corpus, batch_size=hps.batch_size)
    # iterator = get_train_batch('dataset/corpus_train_loop_1.txt', batch_size=hps.batch_size)

    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    summary_writer = tf.summary.FileWriter('tf-log/run-{}-{}-{}'.format(now, mark))

    model = BaseModel(hps, iterator=iterator)
    saver = tf.train.Saver()
    sess = tf.Session()
    if model_path is not None:
        print('load pre-trained')
        saver.restore(sess, save_path=model_path)

    min_loss, min_loss_global_step, min_sess = float('inf'), 0, None

    def save_model(_loss, global_step, _sess):
        model_file_name = './models/step-{}-loss-{}-mark-{}'.format(global_step, _loss, mark)
        saver.save(_sess, model_file_name, global_step=global_step)
        return model_file_name + '-' + str(global_step)

    total_steps = 0

    with sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    loss, _, summary = sess.run([model.loss, model.op, model.summary])
                    summary_writer.add_summary(summary)

                    if total_steps % 100 == 0:
                        print("epoch: {} loss: {}".format(i, loss))
                        summary_writer.flush()
                        if loss < min_loss:
                            min_loss = loss
                            global_steps = sess.run(model.global_steps)
                            min_loss_global_step = global_steps
                            min_sess = sess

                    total_steps += 1

                except tf.errors.OutOfRangeError:
                    break # break while, into another for loop

        global_steps = sess.run(model.global_steps)
        print(global_steps)

        if loss < min_loss:
            min_loss = loss
            min_loss_global_step = global_steps
            min_sess = sess

        print('final loss {} precision is {}'.format(min_loss, np.e ** (-min_loss)))
        model_path = save_model(min_loss, min_loss_global_step, min_sess)
        return model_path


hps1 = Hps(); hps1.hidden_layers = [100, 100]
hps2 = Hps(); hps2.hidden_layers = [60, 60]
hps3 = Hps(); hps3.hidden_layers = [70, 70]
HPS = [hps1, hps2, hps3]
HPS = [hps1]


if __name__ == '__main__':
    # train(HPS[2])

    INIT_CORPUS = 'dataset/cifar-10-batches-py/data_batch_1'

    train_corpus_file = 'dataset/cifar10_new_label_1'
    # train_corpus_file = 'dataset/mini_corpus_train.txt'
    train_corpus_file = 'dataset/cifar-10-batches-py/data_batch_1'

    if train_corpus_file == INIT_CORPUS:
        model_paths = [None] * 3
    else:
        model_paths = [line.strip() for line in open('model_paths.txt')]

    with open('model_paths.txt', 'w') as f:
        models = [train(hps, train_corpus=train_corpus_file, model_path=m_p)
                  for hps, m_p in zip(HPS, model_paths)]
        f.writelines('\n'.join(models))
