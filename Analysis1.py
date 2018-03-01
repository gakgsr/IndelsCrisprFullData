import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


indel_count_matrix = np.loadtxt("indel_count_matrix.txt")
file = open("row_index.txt")
row_index = file.readlines()



[numberofindels,numberoffiles] = np.shape(indel_count_matrix)

number_of_files_per_indel = []
number_of_indels_per_file = []
for i in range(numberofindels):
    number_of_files_per_indel.append(np.count_nonzero(indel_count_matrix[i]))

for i in range(numberoffiles):
    number_of_indels_per_file.append(np.count_nonzero( np.transpose(indel_count_matrix)[i] ) )


print "Number of Indel Types = ", np.size(number_of_files_per_indel)
print "Number of Files = ", np.shape(indel_count_matrix)[1]

print "Max Indel", row_index[np.argmax(number_of_files_per_indel)]
print "Max Indel Occurance = ", max(number_of_files_per_indel)

print "Min Indel", row_index[np.argmin(number_of_files_per_indel)]
print "Min Indel Occurance = ", min(number_of_files_per_indel)

for i in range(20):
    print row_index[np.argsort(number_of_files_per_indel)[::-1][i]].strip("\n")

indel_count_matrix_small = np.copy(indel_count_matrix)
indel_count_matrix_small = indel_count_matrix_small
indel_count_matrix_small = indel_count_matrix_small[np.argsort(number_of_files_per_indel)[::-1][0:200]]
print np.shape(indel_count_matrix_small)

X = TSNE(n_components=2, random_state=0).fit_transform(np.transpose(indel_count_matrix_small))


plt.plot(X[:,0],X[:,1],'ro')
print np.shape(X)
plt.show()



#plt.plot(number_of_files_per_indel)
#plt.xlabel("Indel")
#plt.ylabel("Number of CRISPR Outcomes")

#matplotlib.pyplot.stem(number_of_indels_per_file)
#plt.xlabel("CRISPR Outcome")
#plt.ylabel("Number of Indels")
#plt.show()