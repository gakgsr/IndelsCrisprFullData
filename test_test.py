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
import re
import ot
import time
from scipy.stats import kendalltau




print "loading name_genes_grna_unique ..."
name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))
#name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique.p', 'rb'))
print "loading name_indel_type_unique ..."
name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
print "loading indel_count_matrix ..."
indel_count_matrix = pickle.load(open('storage/indel_count_matrix_one_patient_per_site.p', 'rb'))
#indel_count_matrix = pickle.load(open('storage/indel_count_matrix.p', 'rb'))
print "loading indel_prop_matrix ..."
indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix_one_patient_per_site.p', 'rb'))
#indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix.p', 'rb'))
print "loading length_indel ..."
length_indel_insertion = pickle.load(open('storage/length_indel_insertion.p', 'rb'))
length_indel_deletion = pickle.load(open('storage/length_indel_deletion.p', 'rb'))
print "loading homopolymer matrix"
homopolymer_matrix = pickle.load(open('storage/homopolymer_matrix_w-3:3.p', 'rb'))


def distance_to_transcribed_sequence(name_genes_grna_unique):
    minDistTSS = []
    minDistTSE = []
    disT = pickle.load(open('/Users/amirali/Projects/disTSS_annotations.pkl', 'rb'))
    location_dict = {}
    with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            location_dict[row[0].split(',')[0]] = row[0].split(',')[4]

    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        location = location_dict[site_name_list[1] + '-' + site_name_list[2]]
        minDistTSS.append(np.mean(disT[location[1:]][:,0]))
        minDistTSE.append(np.mean(disT[location[1:]][:,1]))

    return minDistTSS,minDistTSE

minDistTSS,minDistTSE = distance_to_transcribed_sequence(name_genes_grna_unique)
print np.mean(minDistTSS)
print np.mean(minDistTSE)