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




def longest_substring_passing_cutsite(strng,character):
    len_substring=0
    longest=0
    label_set = []
    midpoint = len(strng)/2
    for i in range(len(strng)):
        if i > 1:
            if strng[i] != strng[i-1] or strng[i] != character:
                len_substring = 0
                label_set = []
        if strng[i] == character:
            label_set.append(i)
            len_substring += 1
        if len_substring > longest and (midpoint-1 in label_set or 3 in label_set):
            longest = len_substring

    return longest


def homology_matrix_finder(name_genes_grna_unique):
    # extract genomic context
    context_genome_dict = {}
    spacer_dict = {}
    with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_counter = 0
        for row in spamreader:
            context = row[0].split(',')[6]
            context = context.replace('a', 'A')
            context = context.replace('c', 'C')
            context = context.replace('t', 'T')
            context = context.replace('g', 'G')

            spacer_dict[row[0].split(',')[0]] = row[0].split(',')[2]
            context_genome_dict[row[0].split(',')[0]] = context
    # extract homology matrix
    homology_matrix =  np.zeros((4,len(name_genes_grna_unique)  ))
    site_count = 0
    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        context = context_genome_dict[site_name_list[1] + '-' + site_name_list[2]]
        nuc_count = 0
        for nuc in ['A', 'C', 'G', 'T']:
            homology_matrix[nuc_count,site_count] = int(longest_substring_passing_cutsite(context[50-5:50+5], nuc))
            #print context[50-3:50+3]
            nuc_count+=1
        site_count+=1


    return homology_matrix


#print "loading name_genes_grna_unique ..."
name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))

homology_matrix = homology_matrix_finder(name_genes_grna_unique)
print 'max', np.max(homology_matrix)
print 'min', np.min(homology_matrix)
print homology_matrix
print np.shape(homology_matrix)
pickle.dump(homology_matrix, open('Tcell-files/homology_matrix_UNIQUE.p', 'wb'))

