import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:\\Users\\Jonathan\\Documents\\BGU\\Semester 8\\courses\\DL\\assignments\\1\\logs.csv')

for metric in ['cost', 'accuracy']:
    plt.plot(df['epoch'], df[metric], label='without batchnorm')
    plt.plot(df['epoch'], df['norm_'+metric], label='with batchnorm')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    # plt.title('Convergence progression')

    plt.savefig('C:\\Users\\Jonathan\\Documents\\BGU\\Semester 8\\courses\\DL\\assignments\\1\\'+metric+'.png')
    plt.show()
