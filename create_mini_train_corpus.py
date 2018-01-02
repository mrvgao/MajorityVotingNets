import random

total_number = 50


def func(x, y):
    if y > x: return 1
    else: return 0


with open('dataset/mini_corpus_train.txt', 'w') as f:
    for i in range(total_number):
        random_x = random.randrange(-1000, 1000)
        random_y = random.randrange(-1000, 1000)
        f.write('\n{}\t{}\t{}'.format(random_x, random_y, func(random_x, random_y)))
