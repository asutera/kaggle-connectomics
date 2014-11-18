import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Path to network files if they are not in the same folder
path = ''

n_nodes = 1000

for dataset in ["normal-1", "normal-2", "normal-3",
                "normal-4", "lowcc", "lowcon",
                "highcon", "highcc", "normal-3-highrate",
                "normal-4-lownoise"]:

    name = path + 'network_' + dataset + '.txt'
    raw_graph = np.loadtxt(name, delimiter=",")
    row = raw_graph[:, 0] - 1
    col = raw_graph[:, 1] - 1
    data = raw_graph[:, 2]
    valid_index = data > 0
    net = coo_matrix((data[valid_index], (row[valid_index], col[valid_index])),
                     shape=(n_nodes, n_nodes))

    graph = net.toarray()

    _, singular_values, _ = svd(graph)
    singular_values /= singular_values.sum()
    plt.plot(np.sort(singular_values)[::-1], label=dataset)
plt.legend(loc="best", prop={'size': 12}).draw_frame(False)
plt.ylabel('Singular values', size=12)
plt.xlabel('Components', size=12)
plt.savefig("singular_values_all.pdf", bbox_inches='tight')

plt.close()

for dataset in ["normal-1", "normal-2", "normal-3",
                "normal-4", "lowcc", "lowcon",
                "highcon", "highcc", "normal-3-highrate",
                "normal-4-lownoise"]:

    name = path + 'network_' + dataset + '.txt'
    raw_graph = np.loadtxt(name, delimiter=",")
    row = raw_graph[:, 0] - 1
    col = raw_graph[:, 1] - 1
    data = raw_graph[:, 2]
    valid_index = data > 0
    net = coo_matrix((data[valid_index], (row[valid_index], col[valid_index])),
                     shape=(n_nodes, n_nodes))

    graph = net.toarray()

    clf = PCA(whiten=True)
    clf = clf.fit(graph)
    plt.plot(np.sort(clf.explained_variance_ratio_[::])[::-1], label=dataset)
plt.legend(loc="best", prop={'size': 12}).draw_frame(False)
plt.ylabel('Explained variance ratio', size=12)
plt.xlabel('Components', size=12)

plt.savefig("explained_variance_all.pdf", bbox_inches='tight')
