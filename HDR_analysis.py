import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import entropy

def pears_cor(aa,bb):
  return np.dot( (aa-np.mean(aa))/np.linalg.norm(aa-np.mean(aa) ) , (bb-np.mean(bb))/np.linalg.norm(bb-np.mean(bb) ) )

def entrop_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    entrop = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        entrop.append(entropy(vec))

    return np.asarray(entrop)

def expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_count_matrix)

  exp_insertion_length = np.zeros(site_num,dtype=float)
  exp_deletion_length = np.zeros(site_num,dtype=float)

  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  insertion_only_fraction_matrix =  np.multiply(indel_fraction_mutant_matrix, np.reshape(insertion_indicator,(-1,1)) )
  deletion_only_fraction_matrix = np.multiply(indel_fraction_mutant_matrix,  np.reshape(deletion_indicator,(-1,1)) )

  insertion_only_fraction_matrix = insertion_only_fraction_matrix / np.reshape(np.sum(insertion_only_fraction_matrix, axis=0), (1, -1))
  deletion_only_fraction_matrix = deletion_only_fraction_matrix / np.reshape(np.sum(deletion_only_fraction_matrix, axis=0), (1, -1))


  for site_index in range(site_num):
    exp_insertion_length[site_index] = np.inner(length_indel_insertion,insertion_only_fraction_matrix[:,site_index])
    exp_deletion_length[site_index] = np.inner(length_indel_deletion, deletion_only_fraction_matrix[:, site_index])

  # some sites do not get any insertions. this cuase nan. we make those entries zero.
  for i in range(np.size(exp_insertion_length)):
    if np.isnan(exp_insertion_length[i]):
      exp_insertion_length[i] = 0

  return exp_insertion_length,exp_deletion_length

def fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_count_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)


  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(insertion_indicator,indel_fraction_mutant_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(deletion_indicator, indel_fraction_mutant_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna

def fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_prop_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)


  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(insertion_indicator,indel_prop_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(deletion_indicator, indel_prop_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna

#######################################################################################################################
### HDR
name_genes_grna_unique = pickle.load(open('HDR/name_genes_grna_ALL_matched.p', 'rb'))
name_indel_type_unique = pickle.load(open('HDR/name_indel_type_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('HDR/indel_count_matrix_ALL_matched.p', 'rb'))
no_variant_vec = pickle.load(open('HDR/no_variant_vec_ALL_matched.p', 'rb'))
other_vec = pickle.load(open('HDR/other_vec_ALL_matched.p', 'rb'))
snv_vec = pickle.load(open('HDR/snv_vec_ALL_matched.p', 'rb'))
hdr_vec = pickle.load(open('HDR/hdr_vec_ALL_matched.p', 'rb'))
length_indel_insertion = pickle.load(open('HDR/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('HDR/length_indel_deletion_ALL.p', 'rb'))
spacer_pam_list = pickle.load(open('HDR/spacer_pam_list_ALL_matched.p', 'rb'))
HDR_TCELL_matching_vector = pickle.load(open('HDR/HDR_TCELL_matching_vector.p', 'rb'))


SSS = np.sum(indel_count_matrix,axis=1)
III = np.argsort(SSS)[::-1][0:500]
indel_prop_matrix = indel_count_matrix[III,:] / np.reshape(np.sum(indel_count_matrix[III,:], axis=0)+no_variant_vec, (1, -1))

fraction_insertions_HDR, fraction_deletions_HDR = fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion)
fraction_insertions_all_HDR, fraction_deletions_all_HDR = fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion[III],length_indel_deletion[III])
exp_insertion_length_HDR, exp_deletion_length_HDR = expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion)
entrop_HDR = entrop_finder(indel_count_matrix)
hdr_eff = hdr_vec / (snv_vec+other_vec+no_variant_vec+np.sum(indel_count_matrix,axis=0))
edit_HDR = np.sum(indel_count_matrix,axis=0) / (snv_vec+other_vec+no_variant_vec+np.sum(indel_count_matrix,axis=0))
#######################################################################################################################


### tcells
mean_eff_vec = pickle.load(open('Tcell-files/eff_vec_mean_UNIQUE_no_others.p', 'rb'))
mean_eff_vec = np.asarray(mean_eff_vec)
fraction_insertions = pickle.load(open('Tcell-files/fraction_insertions_UNIQUE.p', 'rb'))
fraction_deletions = pickle.load(open('Tcell-files/fraction_deletions_UNIQUE.p', 'rb'))
fraction_insertions_all = pickle.load(open('Tcell-files/fraction_insertions_all_UNIQUE.p', 'rb'))
fraction_deletions_all = pickle.load(open('Tcell-files/fraction_deletions_all_UNIQUE.p', 'rb'))
exp_insertion_length = pickle.load(open('Tcell-files/exp_insertion_length_UNIQUE.p', 'rb'))
exp_deletion_length = pickle.load(open('Tcell-files/exp_deletion_length_UNIQUE.p', 'rb'))
diversity = pickle.load(open('Tcell-files/diversity_UNIQUE.p', 'rb'))


edit_eff = mean_eff_vec[HDR_TCELL_matching_vector]
fraction_insertions = fraction_insertions[HDR_TCELL_matching_vector]
fraction_deletions = fraction_deletions[HDR_TCELL_matching_vector]
fraction_insertions_all = fraction_insertions_all[HDR_TCELL_matching_vector]
fraction_deletions_all = fraction_deletions_all[HDR_TCELL_matching_vector]
exp_insertion_length = exp_insertion_length[HDR_TCELL_matching_vector]
exp_deletion_length = exp_deletion_length[HDR_TCELL_matching_vector]
diversity = diversity[HDR_TCELL_matching_vector]



print "r2 between original and HDR repair outcomes"
print "Edit efficiency", r2_score(edit_eff,edit_HDR)
print "Fraction of insertion/deletions (mutant)", r2_score(fraction_deletions_HDR,fraction_deletions)
print "Fraction of deletions (total)", r2_score(fraction_insertions_all_HDR,fraction_insertions_all)
print "Fraction of insertions (total)", r2_score(fraction_deletions_all_HDR,fraction_deletions_all)
print "Expected insertion length", r2_score(exp_insertion_length_HDR,exp_insertion_length)
print "Expected deletion length", r2_score(exp_deletion_length_HDR,exp_deletion_length)
print "Diversity", r2_score(entrop_HDR,diversity)



print "-----"
print "Number of outcomes", len(HDR_TCELL_matching_vector)
print "Number of unique sites", len(set(HDR_TCELL_matching_vector))
print "-----"
print "HDR mean =", np.mean(hdr_eff)
print "HDR std", np.std(hdr_eff)
print "HDR max", max(hdr_eff)
print "HDR min", min(hdr_eff)

plt.hist(hdr_eff)
plt.savefig('HDR_plots/HDR_hist.pdf')
plt.clf()

plt.hist(edit_eff)
plt.savefig('HDR_plots/EDIT_hist.pdf')
plt.clf()

plt.plot(edit_eff,hdr_eff,'o')
plt.xlabel('Edit Ef.')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/EDIT_HDR.pdf')
plt.clf()


plt.plot(fraction_insertions,hdr_eff,'o')
plt.xlabel('frac insertion')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/fraction_insertions_HDR.pdf')
plt.clf()

plt.plot(fraction_deletions,hdr_eff,'o')
plt.xlabel('frac deletion')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/fraction_deletions_HDR.pdf')
plt.clf()

plt.plot(fraction_insertions_all,hdr_eff,'o')
plt.xlabel('frac insertion all')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/fraction_insertions_all_HDR.pdf')
plt.clf()

plt.plot(fraction_deletions_all,hdr_eff,'o')
plt.xlabel('frac deletion all')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/fraction_deletions_all_HDR.pdf')
plt.clf()

plt.plot(exp_insertion_length,hdr_eff,'o')
plt.xlabel('expected insertion length')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/exp_insertion_length_HDR.pdf')
plt.clf()

plt.plot(exp_deletion_length,hdr_eff,'o')
plt.xlabel('expected deletion length')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/exp_deletion_length_HDR.pdf')
plt.clf()

plt.plot(diversity,hdr_eff,'o')
plt.xlabel('diversity')
plt.ylabel('HDR Ef.')
plt.savefig('HDR_plots/diversity_HDR.pdf')
plt.clf()



print "-----"
print "Correlation of HDR with Repair outcomes"
print "Pearson Correlation: Edit efficiency / HDR (coefficient,pvalue)"
print pearsonr(hdr_eff,edit_eff)

print "Pearson Correlation: Fraction of insertions (mutant) / HDR"
print pearsonr(hdr_eff,fraction_insertions)

print "Pearson Correlation: Fraction of deletions (mutant) / HDR"
print pearsonr(hdr_eff,fraction_deletions)

print "Pearson Correlation: Fraction of insertions (total) / HDR"
print pearsonr(hdr_eff,fraction_insertions_all)

print "Pearson Correlation: Fractio of deletions (total) / HDR"
print pearsonr(hdr_eff,fraction_deletions_all)

print "Pearson Correlation: Expected insertion length / HDR"
print pearsonr(hdr_eff,exp_insertion_length)

print "Pearson Correlation: Expected deletion length / HDR"
print pearsonr(hdr_eff,exp_deletion_length)

print "Pearson Correlation: Diversity / HDR"
print pearsonr(hdr_eff,diversity)

print "-----"
print "r2 Edit efficiency / HDR", r2_score(hdr_eff,edit_eff)
print "r2 Fraction of insertions (mutant) / HDR", r2_score(hdr_eff,fraction_insertions)
print "r2 Fraction of deletions (mutant) / HDR", r2_score(hdr_eff,fraction_deletions)
print "r2 Fraction of insertions (total) / HDR", r2_score(hdr_eff,fraction_insertions_all)
print "r2 Fraction of insertions (total) / HDR", r2_score(hdr_eff,fraction_deletions_all)
print "r2 Expected insertion length / HDR", r2_score(hdr_eff,exp_insertion_length)
print "r2 Expected deletion length / HDR", r2_score(hdr_eff,exp_deletion_length)
print "r2 Diversity / HDR", r2_score(hdr_eff,diversity)
print "-----"
print "Correlation of HDR with Chromatin factors"

chrom_label_matrix = pickle.load(open('Tcell-files/chrom_label_matrix_UNIQUE.p', 'rb'))
chrom_mat_name = pickle.load(open('storage/chrom_label_dic_name.p', 'rb'))
print np.shape(chrom_label_matrix)
for col in range(33):
  chrom_vec = chrom_label_matrix[:,col]
  chrom_vec = chrom_vec[HDR_TCELL_matching_vector]
  chrom_vec[np.argwhere(np.isnan(chrom_vec))] = np.nanmean(chrom_vec)

  aaa = chrom_vec / np.linalg.norm(chrom_vec)
  bbb = hdr_eff / np.linalg.norm(hdr_eff)

  print chrom_mat_name[col]
  print "r2 = ", r2_score(hdr_eff,chrom_vec)
  print "Pearson Correlation:", pearsonr(hdr_eff,chrom_vec)

  plt.plot(chrom_vec, hdr_eff, 'o')
  plt.xlabel('%s' %chrom_mat_name[col])
  plt.ylabel('HDR Efficiency')
  plt.savefig('HDR_plots/chromatin/%s_HDR.pdf' %chrom_mat_name[col])
  plt.clf()





# aaa = 50
# print hdr_eff[aaa]
# print name_genes_grna_unique[aaa]
# print hdr_vec[aaa]
# print other_vec[aaa]
# print no_variant_vec[aaa]
# print np.sum(indel_count_matrix,axis=0)[aaa]
# iii = np.argmax(indel_count_matrix[:,aaa])
# print name_indel_type_unique[iii]
# print indel_count_matrix[iii,aaa]



