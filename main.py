from basemodel import train
from majority_voting_model import get_precision_of_ensemble
from majority_voting_model import read_unlabel_data
from majority_voting_model import create_new_labled_data
import random
from hyperparamters import HPS
from tqdm import tqdm


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
        if (i % 5 == 0) or (i == max_size - 1): get_precision_of_ensemble()
        dataset_index = random.choice([2, 3, 4, 5])

        initial_trian_size = HPS[0].total_size

        new_data_x, new_data_y = read_unlabel_data(dataset_index, unlabel_test_size=initial_trian_size * 2)
        train_corpus = create_new_labled_data(new_data_x, new_data_y, HPS[0].total_size, loop=i)
        train_corpus_f = train_corpus


if __name__ == '__main__':
    train_with_loop(10000)
