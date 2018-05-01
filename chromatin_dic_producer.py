import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import csv


name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))

chrom_mat = pickle.load(open('/Users/amirali/Projects/CADD_annotations.pkl', 'rb'))
chrom_label_dic_name = chrom_mat['name_columns']

location_dict = {}
with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    row_counter = 0
    for row in spamreader:
        location_dict[row[0].split(',')[0]] = row[0].split(',')[4]

chrom_label_dic = {}

for site_name in name_genes_grna_unique:
    site_name_list = site_name.split('-')
    location = location_dict[site_name_list[1] + '-' + site_name_list[2]]
    #print location[1:]
    chrom_label_dic[site_name] = chrom_mat[location[1:]]


pickle.dump(chrom_label_dic, open('storage/chrom_label_dic.p', 'wb'))
pickle.dump(chrom_label_dic_name, open('storage/chrom_label_dic_name.p', 'wb'))