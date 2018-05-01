import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import csv
import math
import glob
import math
import random
from scipy import cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


alllist=[]
file=open('/Users/amirali/Projects/spacer_pam_exon.txt','r')
for line in file:
    alllist.append(line[14:20])

listset = set(alllist)
uniquelist =  list(set(alllist))
print "number of possible sites that reside on exons =", len(alllist)
print "number of possible spacer+pam that reside on exon =", len(listset)


max_spacer_to_plot = 300
spacer_distance = np.zeros((max_spacer_to_plot,max_spacer_to_plot))
for i in range(max_spacer_to_plot):
    for j in range(max_spacer_to_plot):
        #spacer_distance[i,j] = levenshteinDistance(uniquelist[i],uniquelist[j])
        spacer_distance[i, j] = hamming(list(uniquelist[i]), list(uniquelist[j]))
        #hamming




kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(spacer_distance)
labels = kmeans.labels_
sort_index = np.argsort(labels)

spacer_distance_sorted = spacer_distance[:, sort_index][sort_index]


fig, axis = plt.subplots()
heat_map = axis.pcolor(spacer_distance_sorted, cmap=plt.cm.Greens)
#axis.set_yticks(np.arange(PP.shape[0])+0.5, minor=False)
#axis.set_xticks(np.arange(PP.shape[1])+0.5, minor=False)
#axis.invert_yaxis()
#axis.set_yticklabels(['A','T','C','G'], minor=False)
#axis.set_xticklabels(['A','T','C','G'], minor=False)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.xlabel('Neighboring Nucleotide')
#plt.ylabel('Inserted Nucleotide')
plt.colorbar(heat_map)
plt.savefig('plots/Exon_all_spacers_distance_matrix.pdf')
plt.clf