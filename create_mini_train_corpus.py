import random
import matplotlib.pyplot as plt
import numpy as np

total_number = 50


def func(x, y):
    new_x = x + int(np.random.normal(20, 1))
    new_y = y + int(np.random.normal(20, 1))
    if y > abs(x) or y < -abs(x):
        return (new_x, new_y), 1
    else:
        return (new_x, new_y), 0


type_1 = []
type_2 = []


def create_data():
    with open('dataset/mini_corpus_train.txt', 'w') as f:
        for i in range(total_number):
            random_x = random.randrange(-1000, 1000)
            random_y = random.randrange(-1000, 1000)
            x_y, _type = func(random_x, random_y)
            if _type == 0: type_1.append(x_y)
            elif _type == 1: type_2.append(x_y)

            x, y = x_y
            f.write('\n{}\t{}\t{}'.format(x, y, _type))

    plt.scatter(*zip(*type_2), color='g')
    plt.scatter(*zip(*type_1), color='r')
    plt.show()


if __name__ == '__main__':
    create_data()
