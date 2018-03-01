from collections import Counter


list = []
with open("sequence_pam_gene_grna.csv") as f:
    for line in f:
        list.append(line.split(',')[3].strip('\r\n'))

c=Counter(list)
print c.values()
print c.keys()