class Hps:
    epoch = 30
    batch_size = 256
    x_size = 32*32*3
    y_size = 10
    learning_rate = 1e-3
    regularization = 1e-3
    hidden_layers = [7, ]


hps1 = Hps(); hps1.hidden_layers = [100, 100]
hps2 = Hps(); hps2.hidden_layers = [80, 80]
hps3 = Hps(); hps3.hidden_layers = [130, 130]
HPS = [hps1, hps2, hps3]
