from preprocess_indel_files import preprocess_indel_files
import pickle
import numpy as np

data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"

# name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix, no_variant_vec, other_vec, snv_vec, length_indel_insertion, length_indel_deletion = preprocess_indel_files(data_folder)
#
# print np.shape(name_genes_unique)
# print np.shape(name_genes_grna_unique)
# print np.shape(name_indel_type_unique)
# print np.shape(indel_count_matrix)
# print np.shape(indel_prop_matrix)
# print np.shape(no_variant_vec)
# print np.shape(other_vec)
# print np.shape(snv_vec)
# print np.shape(length_indel_insertion)
# print np.shape(length_indel_deletion)
#
# pickle.dump(name_genes_unique, open('storage/name_genes_unique_BIG.p', 'wb'))
# pickle.dump(name_genes_grna_unique, open('storage/name_genes_grna_unique_BIG.p', 'wb'))
# pickle.dump(name_indel_type_unique, open('storage/name_indel_type_unique_BIG.p', 'wb'))
# pickle.dump(indel_count_matrix, open('storage/indel_count_matrix_BIG.p', 'wb'))
# pickle.dump(indel_prop_matrix, open('storage/indel_prop_matrix_BIG.p', 'wb'))
# pickle.dump(length_indel_insertion, open('storage/length_indel_insertion_BIG.p', 'wb'))
# pickle.dump(length_indel_deletion, open('storage/length_indel_deletion_BIG.p', 'wb'))
# pickle.dump(no_variant_vec, open('storage/no_variant_vec_BIG.p', 'wb'))
# pickle.dump(other_vec, open('storage/other_vec_BIG.p', 'wb'))
# pickle.dump(snv_vec, open('storage/snv_vec_BIG.p', 'wb'))

# name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_BIG.p', 'rb'))
# name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique_BIG.p', 'rb'))
# indel_count_matrix = pickle.load(open('storage/indel_count_matrix_BIG.p', 'rb'))
# no_variant_vec = pickle.load(open('storage/no_variant_vec_BIG.p', 'rb'))
# other_vec = pickle.load(open('storage/other_vec_BIG.p', 'rb'))
# snv_vec = pickle.load(open('storage/snv_vec_BIG.p', 'rb'))
# length_indel_insertion = pickle.load(open('storage/length_indel_insertion_BIG.p', 'rb'))
# length_indel_deletion = pickle.load(open('storage/length_indel_deletion_BIG.p', 'rb'))
#
# eff_vec = (np.sum(indel_count_matrix,axis=0) + other_vec) / (np.sum(indel_count_matrix,axis=0)+no_variant_vec+snv_vec)
# pickle.dump(eff_vec, open('storage/eff_vec_BIG_others_in_numinator.p', 'wb'))

########

# name_genes_grna_BIG = pickle.load(open('storage/name_genes_grna_unique_BIG.p', 'rb'))
# name_genes_grna_Tcell = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
# eff_vec_BIG = pickle.load(open('storage/eff_vec_BIG_others_in_numinator.p', 'rb'))
#
# eff_vec = []
# for file in name_genes_grna_Tcell:
#     print file
#     eff_vec.append(eff_vec_BIG[np.where(np.asarray(name_genes_grna_BIG)==file)[0]][0] )
#
#
# print eff_vec
# print np.shape(eff_vec)
# pickle.dump(eff_vec, open('Tcell-files/my_eff_vec_UNIQUE_others_in_numinator.p', 'wb'))