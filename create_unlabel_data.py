import random

with open('dataset/unlabel_data.txt', 'w') as f:
    for i in range(50000):
        f.write('\n{}'.format(random.randrange(-5000, 5000)))