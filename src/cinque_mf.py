from logistic_mf.logistic_mf import LogisticMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import csv
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "../data/cinque_stats.csv"
NUM_FACTORS = 2

language_indices = []
count_data = []

with open(INPUT_FILE, 'r') as f:
    reader = csv.reader(f)
    headers = reader.__next__()
    order_indices = headers[2:]

    for row in reader:
        language_indices.append(row[1])
        count_data.append([float(r) for r in row[2:]])

counts = np.array(count_data).transpose()

mf = LogisticMF(counts, NUM_FACTORS)
mf.train_model()
mf.print_vectors()

lines = []

with open("../data/logmf-item-vecs-{}".format(NUM_FACTORS), 'r') as f:
    for line in f:
        values = line.split('\t')[1].rstrip().split(' ')
        lines.append([float(i) for i in values])

lines = np.array(lines)

# Do a PCA to 2 dimensions
# pca = PCA(2)
# reduced_matrix = pca.fit_transform(lines)

# Make the labels a bit more manageable for plotting
#order_labels = [o.split("_")[0] for o in order_indices]
order_labels = language_indices
reduced_matrix = lines
#order_labels = order_indices

# Plot the PCA
plt.figure()
for i, label in enumerate(order_labels):
    x, y = reduced_matrix[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        textcoords='offset points',
        ha='right',
        va='bottom'
    )
plt.xlabel("PC1")
plt.xlabel("PC2")
plt.title("PCA of Cinque language orderings")
plt.savefig("../data/ordering_pca.png")

dim = len(order_indices)
cos_matrix = np.zeros((dim, dim))
for i, order1 in enumerate(order_indices):
    for j in range(0, i):
        order2 = order_indices[j]
        cos_matrix[i][j] = max(cosine_similarity(
            lines[order_indices.index(order1)].reshape(1, -1),
            lines[order_indices.index(order2)].reshape(1, -1)
        ), 0)

with open('../data/cos_matrix_{}.csv'.format(NUM_FACTORS), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    writer.writerow([""] + order_indices)
    for i, order in enumerate(order_indices):
        writer.writerow([order] + cos_matrix[i].tolist()) 
