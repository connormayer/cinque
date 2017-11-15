import csv
import matplotlib.pyplot as plt
import numpy as np
import Orange

from copy import deepcopy
from orangecontrib.associate.fpgrowth import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

INPUT_FILE = "../data/cinque_stats.csv"
ORANGE_TABLE = "../data/cinque.basket"
FREQ_PATTERN_FILE = "../data/cinque_freq_patterns.csv"

SUPPORT_THRESHOLD = 0.001
CONFIDENCE_THRESHOLD = 0.005

class CinqueFrequentPatternAnalyzer():
    order_indices = []
    language_indices = []
    count_data = []

    def write_orange_table(self, matrix):
        orange_table = []
        for i, row in enumerate(matrix):
            orange_row = []
            for j, entry in enumerate(row):
                if entry == 1:
                    orange_row.append(self.order_indices[j])
            orange_table.append(orange_row)

        with open(ORANGE_TABLE, 'w') as f:
            for row in orange_table:
                print(','.join(row), file=f)

    def do_frequent_pattern_analysis(self):
        with open(INPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            headers = reader.__next__()
            self.order_indices = headers[2:]

            for row in reader:
                self.language_indices.append(row[1])
                self.count_data.append([int(r) for r in row[2:]])

        self.write_orange_table(self.count_data)

        data = Orange.data.Table(ORANGE_TABLE)
        X, mapping = OneHot.encode(data, include_class=True)
        itemsets = dict(frequent_itemsets(X, SUPPORT_THRESHOLD))
        rules = association_rules(itemsets, CONFIDENCE_THRESHOLD)
        names = {
            item: '{}={}'.format(var.name, val)
            for item, var, val in OneHot.decode(mapping, data, mapping)
        }
        with open(FREQ_PATTERN_FILE, 'w') as f:
            print("{},{},{}".format("Rule", "Support", "Confidence"), file=f)
            for ante, cons, supp, conf in sorted(rules, key=lambda x: -x[2]): 

                print("{} --> {}, {}, {}".format(
                        ' & '.join(names[i] for i in ante),
                        ' & '.join(names[i] for i in cons),
                        supp, conf
                    ),
                    file=f
                )

if __name__ == "__main__":
    analyzer = CinqueFrequentPatternAnalyzer()
    analyzer.do_frequent_pattern_analysis()
