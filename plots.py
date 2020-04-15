import matplotlib.pyplot as plt
import pandas as pd

dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Semester 8\\courses\\DL\\assignments\\1\\'
for metric in ['validation cost', 'validation accuracy']:
    for model in ['normal', 'batchnorm', 'dropout']:
        df = pd.read_csv('%slog_%s.csv' % (dir, model))
        plt.plot(df['epoch'], df[metric], label=model)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.savefig('%s%s.png' % (dir, metric))
    plt.show()
