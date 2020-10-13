import csv
import numpy as np
import matplotlib.pyplot as plt

def readCsv(csv_path):
    x = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\n')
        for line in reader:
            curr_x = float(line[0])
            x.append(curr_x)
    return x

def plotDataset(dataset, title=None, save_to=None, threshold=-1):
    fig = plt.figure()
    plt.grid(True)

    data = []
    labels = []
    for d in dataset:
        x = readCsv(d[1])
        percentage = sum(i > threshold for i in x)/float(len(x)) *100.0
        data.append(x)
        labels.append(d[0] + " - total: {0} ({1}%)".format(len(x), round(percentage,2)))

    n, bins, patches = plt.hist(data, bins=30, label=labels, alpha=1.0, density=True)
    plt.xlim(max(threshold, 0), bins[-1])
    first_bin = next(i for i,v in enumerate(bins) if v > threshold)
    max_y = max([max(i[first_bin:]) for i in n])
    plt.ylim(0, max_y)
    plt.xlabel('VSD')
    plt.ylabel('% of cases')
    plt.legend(loc='upper right')
    axes = plt.gca()
    if(title is not None):
        plt.title(title)
    if(save_to is None):
        plt.show()
    else:
        fig.savefig(save_to, dpi=fig.dpi)

# Object 28
# data = []
# data.append(('l2-pose-loss','./obj28/1k-dataset/vsd-pose-1k.csv'))
# data.append(('l1-abs-depth-loss','./obj28/1k-dataset/vsd-depth-1k.csv'))
# plotDataset(data,
#             title='Object 28 T-LESS - 1k dataset',
#             save_to='./obj28/plot-1k.png')

# data = []
# data.append(('l2-pose-loss','./obj28/5k-dataset/vsd-pose-5k.csv'))
# data.append(('l1-abs-depth-loss','./obj28/5k-dataset/vsd-depth-5k.csv'))
# plotDataset(data,
#             title='Object 28 T-LESS - 5k dataset',
#             save_to='./obj28/plot-5k.png')

# data = []
# data.append(('l2-pose-loss','./obj28/5k-dataset/vsd-pose-5k.csv'))
# data.append(('l1-abs-depth-loss','./obj28/5k-dataset/vsd-depth-5k.csv'))
# plotDataset(data,
#             title='Object 28 T-LESS - 5k dataset',
#             save_to='./obj28/plot-5k-thresholded.png',
#             threshold=0.3)

data = []
data.append(('l2-pose-loss','./obj28/aug-5k-dataset/vsd-pose-5k-aug.csv'))
data.append(('l1-abs-depth-loss','./obj28/aug-5k-dataset/vsd-depth-5k-aug.csv'))
data.append(('pose-then-depth','./obj28/aug-5k-dataset/vsd-pose-then-depth-5k-aug.csv'))
plotDataset(data,
            title='Object 28 T-LESS - 5k dataset augmented',
            save_to='./obj28/plot-5k-aug.png')

data = []
data.append(('l2-pose-loss','./obj28/aug-5k-dataset/vsd-pose-5k-aug.csv'))
data.append(('l1-abs-depth-loss','./obj28/aug-5k-dataset/vsd-depth-5k-aug.csv'))
data.append(('pose-then-depth','./obj28/aug-5k-dataset/vsd-pose-then-depth-5k-aug.csv'))
plotDataset(data,
            title='Object 28 T-LESS - 5k dataset augmented',
            save_to='./obj28/plot-5k-aug-thresholded.png',
            threshold=0.3)

# Object 19
data = []
data.append(('l2-pose-loss','./obj19/aug-5k-dataset/vsd-pose-5k-aug.csv'))
data.append(('l1-abs-depth-loss','./obj19/aug-5k-dataset/vsd-depth-5k-aug.csv'))
data.append(('pose-then-depth','./obj19/aug-5k-dataset/vsd-pose-then-depth-5k-aug.csv'))
plotDataset(data,
            title='Object 19 T-LESS - 5k dataset augmented',
            save_to='./obj19/plot-5k-aug.png')

data = []
data.append(('l2-pose-loss','./obj19/aug-5k-dataset/vsd-pose-5k-aug.csv'))
data.append(('l1-abs-depth-loss','./obj19/aug-5k-dataset/vsd-depth-5k-aug.csv'))
data.append(('pose-then-depth','./obj19/aug-5k-dataset/vsd-pose-then-depth-5k-aug.csv'))
plotDataset(data,
            title='Object 19 T-LESS - 5k dataset augmented',
            save_to='./obj19/plot-5k-aug-thresholded.png',
            threshold=0.3)
