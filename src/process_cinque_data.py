import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

INPUT_FILE = "../data/cinque_stats.csv"

class CinqueProcessor():
    order_indices = []
    language_indices = []
    count_data = []

    def process_cinque_data(self):
        with open(INPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            headers = reader.__next__()
            self.order_indices = headers[2:]

            for row in reader:
                self.language_indices.append(row[1])
                self.count_data.append([int(r) for r in row[2:]])

        count_array = np.array(self.count_data).transpose()

        # Normalize each row to length 1
        row_sums = count_array.sum(axis=1)
        normalized_matrix = count_array / row_sums[:, np.newaxis]
        normalized_matrix[np.isnan(normalized_matrix)] = 0

        # Do a PCA to 2 dimensions
        pca = PCA(2)
        reduced_matrix = pca.fit_transform(normalized_matrix)

        # Make the labels a bit more manageable for plotting
        order_labels = [o.split("_")[0] for o in order_indices]

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
        plt.savefig("ordering_pca.png")

if __name__ == "__main__":
    proc = CinqueProcessor()
    proc.process_cinque_data()
