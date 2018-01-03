import matplotlib.pyplot as plt

data = [float(line.strip().replace('precision: ', '')) for line in open('precision_log') if line.strip() != '']

plt.plot(data)
plt.show()