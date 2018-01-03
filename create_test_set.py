from create_mini_train_corpus import func
import random

total_number = 10000


with open('dataset/test.txt', 'w') as f:
    for i in range(total_number):
        random_x = random.randrange(-10000, 10000)
        random_y = random.randrange(-10000, 10000)
        x_y, _type = func(random_x, random_y)
        x, y = x_y
        f.write('\n{}\t{}\t{}'.format(x, y, _type))