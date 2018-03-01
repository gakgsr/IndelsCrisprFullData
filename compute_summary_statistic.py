from preprocess_indel_files import preprocess_indel_files
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix):
  # Compute most common indels

  #number_of_files_per_indel = []
  #for i in range(indel_count_matrix.shape[0]):
  #  number_of_files_per_indel.append(np.count_nonzero(indel_count_matrix[i]))
  #print "The twenty most commonly occurring indels are:"
  #for i in range(20):
  #  print name_indel_type_unique[np.argsort(number_of_files_per_indel)[::-1][i]]

  #
  # Plot TSNE
  #indel_count_matrix_small = np.copy(indel_count_matrix)
  #indel_count_matrix_small = indel_count_matrix_small[np.argsort(number_of_files_per_indel)[::-1][0:200]]
  #X = TSNE(n_components=2, random_state=0).fit_transform(np.transpose(indel_count_matrix_small))
  #plt.scatter(X[:, 0], X[:, 1])
  #plt.savefig('all-genes-tsne.pdf')
  #plt.clf()

  # Plot Full PCA
  #pca = PCA(n_components=2)
  #X = pca.fit_transform(indel_count_matrix)
  #plt.scatter(X[:, 0], X[:, 1])
  #plt.savefig('all-genes-pca-non-normalized.pdf')
  #plt.clf()

  # Plot normalized PCA
  #indel_count_matrix_copy = np.array(np.copy(indel_count_matrix), dtype = float)
  #indel_count_matrix_copy = indel_count_matrix_copy/np.reshape(np.sum(indel_count_matrix_copy, axis = 0), (1, -1))
  #indel_count_matrix_copy -= np.reshape(np.mean(indel_count_matrix_copy, axis = 1), (-1, 1))
  #indel_count_matrix_copy = indel_count_matrix_copy/np.reshape(np.linalg.norm(indel_count_matrix_copy, axis = 0), (1, -1))
  #pca = PCA(n_components = 2)
  #X = pca.fit_transform(indel_count_matrix_copy)
  #plt.scatter(X[:, 0], X[:, 1])
  #plt.xlim(-0.1, 2)
  #plt.ylim(-1, 1)
  #plt.savefig('all-genes-pca-normalized.pdf')
  #plt.clf()

  # Another normalized PCA, on proportions
  #indel_prop_matrix_copy = np.array(np.copy(indel_prop_matrix), dtype = float)
  #indel_prop_matrix_copy -= np.reshape(np.mean(indel_prop_matrix_copy, axis = 1), (-1, 1))
  #indel_prop_matrix_copy = indel_prop_matrix_copy/np.reshape(np.linalg.norm(indel_prop_matrix_copy, axis = 0), (1, -1))
  #pca = PCA(n_components = 2)
  #X = pca.fit_transform(indel_prop_matrix_copy)
  #plt.scatter(X[:, 0], X[:, 1])
  #plt.xlim(-0.1, 2)
  #plt.ylim(-1, 1)
  #plt.savefig('all-genes-pca-normalized-prop.pdf')
  #plt.clf()

  '''
  # Plot heat map of cosine distances
  heat_map_inner_prod = np.matmul(np.transpose(indel_count_matrix_small), indel_count_matrix_small)
  # The next 2 lines change the plot from inner product to cosine distance
  #  For just the inner product, comment these two lines
  col_wise_norms = np.expand_dims(np.linalg.norm(indel_count_matrix_small, axis = 0), axis = 1)
  heat_map_inner_prod = np.divide(heat_map_inner_prod, np.matmul(col_wise_norms, np.transpose(col_wise_norms)))
  fig, axis = plt.subplots()
  heat_map = axis.pcolor(heat_map_inner_prod, cmap = plt.cm.Blues)
  axis.set_yticks(np.arange(heat_map_inner_prod.shape[0])+0.5, minor=False)
  axis.set_xticks(np.arange(heat_map_inner_prod.shape[1])+0.5, minor=False)
  axis.invert_yaxis()
  axis.set_yticklabels(name_indel_type_unique, minor=False)
  axis.set_xticklabels(name_indel_type_unique, minor=False)
  plt.xticks(fontsize=2.5, rotation=90)
  plt.yticks(fontsize=2.5)
  plt.colorbar(heat_map)
  plt.savefig('inner_product_heat_map.pdf')
  plt.clf()
  '''
  ##
  # Threshold indels for each column
  indel_count_matrix_threshold = indel_count_matrix/np.reshape(np.sum(indel_count_matrix, axis = 0), (1, -1))
  indel_count_matrix_threshold[indel_count_matrix_threshold == np.inf] = 0.0
  indel_count_matrix_threshold[indel_count_matrix_threshold == -np.inf] = 0.0
  ins_location = np.zeros(indel_count_matrix.shape, dtype = bool)
  del_locaion = np.zeros(indel_count_matrix.shape, dtype = bool)
  for i in range(len(name_indel_type_unique)):
    if name_indel_type_unique[i].find('I') != -1:
      ins_location[i, :] = 1
    if name_indel_type_unique[i].find('D') != -1:
      del_locaion[i, :] = 1
  threshold_ins = 0.115
  indel_count_matrix_threshold[np.logical_and(indel_count_matrix_threshold <= threshold_ins, ins_location)] = 0.0
  threshold_del = 0.144
  indel_count_matrix_threshold[np.logical_and(indel_count_matrix_threshold <= threshold_del, del_locaion)] = 0.0
  #
  # For each gene-grna pair, count the number of indels above threshold
  count_insertions_gene_grna = np.zeros(len(name_genes_grna_unique), dtype = int)
  count_deletions_gene_grna = np.zeros(len(name_genes_grna_unique), dtype = int)
  for i in range(len(name_genes_grna_unique)): # file or col
    for j in range(indel_count_matrix_threshold.shape[0]): # row or indel
      if indel_count_matrix_threshold[j][i] > 0:
        if name_indel_type_unique[j].find('I') != -1:
          count_insertions_gene_grna[i] += 1
        if name_indel_type_unique[j].find('D') != -1:
          count_deletions_gene_grna[i] += 1
  print "Number of zeros in insertions is %d" % np.sum(count_insertions_gene_grna == 0)
  print "Total number of files is %d" %np.size(count_insertions_gene_grna)
  print "Number of zeros in deletions is %d" % np.sum(count_deletions_gene_grna == 0)
  print "Total number of files is %d" % np.size(count_deletions_gene_grna)

  # Save the output
  indel_family_count_gene_grna = open('indel_family_count_gene_grna.txt', 'w')
  indel_family_count_gene_grna.write("gene_grna_name,insertion_count,deletion_count\n")
  for i in range(len(name_genes_grna_unique)):
    indel_family_count_gene_grna.write("%s,%d,%d\n" % (name_genes_grna_unique[i], count_insertions_gene_grna[i], count_deletions_gene_grna[i]))

  return count_insertions_gene_grna, count_deletions_gene_grna