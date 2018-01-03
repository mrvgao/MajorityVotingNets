from functools import lru_cache


@lru_cache(maxsize=10)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    batch_1 = unpickle('dataset/cifar-10-batches-py/data_batch_1')

    X = batch_1[b'data']


    total_loaded = 0


    def change_to_string(x): return " ".join(map(str, x))


    with open('dataset/cifar10_init_train.txt', 'w') as f:
        print('loading init!')
        for x, y in zip(batch_1[b'data'], batch_1[b'labels']):
            x = change_to_string(x)
            print(total_loaded); total_loaded += 1
            f.write('\n{}\t{}'.format(y, x))


    with open('dataset/cifar10_unlable.txt', 'w') as f:
        for i in range(2, 6):
            data = unpickle('dataset/cifar-10-batches-py/data_batch_{}'.format(i))
            print('loading {}'.format(i))
            for x in data[b'data']:
                x = change_to_string(x)
                print(total_loaded); total_loaded += 1
                f.write('\n{}'.format(x))


    with open('dataset/cifar10_test_set.txt', 'w') as f:
        data = unpickle('dataset/cifar-10-batches-py/test_batch')
        for x, y in zip(data[b'data'], data[b'labels']):
            x = change_to_string(x)
            f.write('\n{}\t{}'.format(y, x))
            print(total_loaded); total_loaded += 1

    print('load finish!')
