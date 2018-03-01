import numpy as np
import glob
import openpyxl
import xlrd

count_data_folder = '/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/counts/'
helper_sheet = '/Users/amirali/Projects/CRISPR-data-Feb18/MasterWellsFIN.xlsx'

writefile = open('sequence_pam_gene_grna_big_file.csv','w')

book = openpyxl.load_workbook(helper_sheet)
sheet = book.active
all_exps=[]
for row in range(2, 6147):
    cell_value = str(sheet['A%d'%row].value)
    cell_value = cell_value.replace('_','-')[6:]
    all_exps.append(cell_value)

for each_file in glob.glob(count_data_folder + "counts-*.txt"):
    f = open(each_file,'r')
    lines = f.readlines()
    f.close()
    line = lines[0]
    line = line.replace('\n', '')
    line = line.replace('_', '-')
    line = line.replace('"', '')
    l = line.split(',')
    curr_gene_name = each_file[len(count_data_folder) + 7:-4].split('-')[0]
    for patient in range(np.size(l)):
        a =  "%s-%s" % (l[patient].split('-')[1], l[patient].split('-')[2])
        #print a
        #print all_exps.index(a)
        guide,well= str(sheet['E%d'% (all_exps.index(a)+2)].value).split('_')
        #print "guide", guide
        #print "well", well
        workbook = xlrd.open_workbook('/Users/amirali/Projects/CRISPR-data/Plate Maps/cr%d.xls' %int(guide))
        worksheet = workbook.sheet_by_name('Cherry-pick crRNA Library Plate')
        #print worksheet.cell(0, 0).value
        well_list = worksheet.col_values(1)
        spacer_list = worksheet.col_values(5)
        pam_list = worksheet.col_values(6)
        #print a,curr_gene_name,spacer_list[well_list.index(well)],pam_list[well_list.index(well)]
        writefile.write('%s,%s,%s,%s\n' %(a,curr_gene_name,spacer_list[well_list.index(well)],pam_list[well_list.index(well)]))


