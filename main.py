from basemodel import train
from majority_voting_model import get_precision_of_ensemble
from majority_voting_model import read_unlabel_data
import random
from load_cifar_10 import unpickle
from hyperparamters import HPS


INIT_CORPUS = 'dataset/cifar-10-batches-py/data_batch_1'


def train_three_models_one_time(train_corpus):
    train_corpus_file = train_corpus

    if train_corpus_file == INIT_CORPUS:
        model_paths = [None] * 3
    else:
        model_paths = [line.strip() for line in open('model_paths.txt')]

    with open('model_paths.txt', 'w') as f:
        models = []
        model_index = 1

        for hps, m_p in zip(HPS, model_paths):
            print('MODEL {}'.format(model_index)); model_index += 1
            path = train(hps, train_corpus=train_corpus_file, model_path=m_p)
            models.append(path)

        f.writelines('\n'.join(models))


def train_with_loop(max_size=20):
    train_corpus_f = INIT_CORPUS

    for i in range(max_size):
        train_three_models_one_time(train_corpus_f)
        print('LOOPING -- {} -- '.format(i))
        get_precision_of_ensemble()
        dataset_index = random.choice([2, 3, 4, 5])

        init_train_set = unpickle(INIT_CORPUS)
        init_x, init_y = init_train_set[b'data'], init_train_set[b'labels']
        train_corpus = read_unlabel_data((init_x, init_y), dataset_index, i)
        train_corpus_f = train_corpus


if __name__ == '__main__':
    train_with_loop(10)
