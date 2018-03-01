import numpy as np

#a = [15,18,17]
#b = [13,8,9]

a = [5,16,14]
b = [23,10,12]

total_rand_baseline = 0
for fold in range(3):
    vec = np.array(a[fold]*[1]+b[fold]*[0])
    rand_baseline = 0

    for repeat in range(10000):
        rand_vec =  np.random.randint(2, size=a[fold]+b[fold])
        rand_baseline = rand_baseline + sum(rand_vec==vec)/float(a[fold]+b[fold])

    rand_baseline = rand_baseline / 10000
    total_rand_baseline = total_rand_baseline + rand_baseline

total_rand_baseline = total_rand_baseline / 3
print total_rand_baseline