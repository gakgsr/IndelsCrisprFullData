import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import csv

# ## this code extracts chromatin features from other cell types

gene_name_micheal = pickle.load(open('/Users/amirali/Projects/hct1_gene_names.pkl', 'rb'))
chrom_mat = pickle.load(open('/Users/amirali/Projects/hct1_chromatin_features.pkl', 'rb'))
all_file_names = pickle.load(open('storage_other_cell/all_file_name.p', 'rb'))

other_cell_gene_list = []
chrom_label_matrix = np.zeros((np.shape(all_file_names)[0],33))
for name in all_file_names:
    other_cell_gene_list.append(gene_name_micheal[name.split('_')[1]])

print len(other_cell_gene_list)
print len(set(other_cell_gene_list))

for counter, name in enumerate(all_file_names):
    chrom_label_matrix[counter, :] = np.nanmean(chrom_mat[name.split('_')[1]],axis=0)

for i in range(np.shape(chrom_label_matrix)[0]):
    for j in range(np.shape(chrom_label_matrix)[1]):
        if np.isnan(chrom_label_matrix[i][j]):
            chrom_label_matrix[i][j] = np.nanmean(chrom_label_matrix[:,j])


pickle.dump(other_cell_gene_list, open('storage_other_cell/other_cell_gene_list.p', 'wb'))
pickle.dump(chrom_label_matrix, open('storage_other_cell/chrom_label_matrix.p', 'wb'))



# ## this code extracts chromatin features from tcell data
# name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
# spacers = pickle.load(open('Tcell-files/spacer_pam_list_UNIQUE.p', 'rb'))
#
# chrom_mat = pickle.load(open('/Users/amirali/Projects/CADD_annotations.pkl', 'rb'))
# chrom_label_dic_name = chrom_mat['name_columns']
#
# location_dict = {}
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     row_counter = 0
#     for row in spamreader:
#         location_dict[row[0].split(',')[0]] = row[0].split(',')[4]
#
# chrom_label_dic = {}
# chrom_label_matrix = np.zeros((np.shape(name_genes_grna_unique)[0],33))
#
# for site_count,site_name in enumerate(name_genes_grna_unique):
#     site_name_list = site_name.split('-')
#     location = location_dict[site_name_list[1] + '-' + site_name_list[2]]
#     #print location[1:]
#     chrom_label_dic[site_name] = np.nanmean(chrom_mat[location[1:]],axis=0)
#     chrom_label_matrix[site_count,:] =  np.nanmean(chrom_mat[location[1:]],axis=0)
#
#
#
# print np.shape(chrom_label_matrix)
#
# #pickle.dump(chrom_label_matrix, open('Tcell-files/chrom_label_matrix_UNIQUE.p', 'wb'))
# #pickle.dump(chrom_label_dic_name, open('storage/chrom_label_dic_name.p', 'wb'))
