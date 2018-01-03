from hyperparamters import Hps
from basemodel import train
from majority_voting_model import get_precision_of_ensemble
from majority_voting_model import read_unlabel_data
import random

hps1 = Hps(); hps1.hidden_layers = [100, 100]
hps2 = Hps(); hps2.hidden_layers = [80, 80]
hps3 = Hps(); hps3.hidden_layers = [130, 130]
HPS = [hps1, hps2, hps3]
# HPS = [hps1]

INIT_CORPUS = 'dataset/cifar-10-batches-py/data_batch_1'


def train_three_models_one_time(train_corpus):
    train_corpus_file = train_corpus

    if train_corpus_file == INIT_CORPUS:
        model_paths = [None] * 3
    else:
        model_paths = [line.strip() for line in open('model_paths.txt')]

    with open('model_paths.txt', 'w') as f:
        models = [train(hps, train_corpus=train_corpus_file, model_path=m_p)
                  for hps, m_p in zip(HPS, model_paths)]
        f.writelines('\n'.join(models))


def train_with_loop(max_size=20):
    train_corpus_f = INIT_CORPUS

    for i in range(max_size):
        train_three_models_one_time(train_corpus_f)
        print('LOOPING -- {} -- '.format(i))
        get_precision_of_ensemble()
        dataset_index = random.choice([2, 3, 4, 5])
        train_corpus = read_unlabel_data(dataset_index, i)
        train_corpus_f = train_corpus


if __name__ == '__main__':
    train_with_loop(10)
