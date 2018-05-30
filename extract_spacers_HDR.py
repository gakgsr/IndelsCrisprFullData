import numpy as np
import glob
import openpyxl
import xlrd
import csv
import warnings
warnings.filterwarnings("ignore")

count_data_folder = '/Users/amirali/Projects/HDR/HDR/'
#helper_sheet = '/Users/amirali/Projects/CRISPR-data-Feb18/MasterWellsFIN.xlsx'
helper_sheet = '/Users/amirali/Projects/HDR/MasterWellsFINv2-updt_withHDR.xlsx'
genomic_refrence_sheet = '/Users/amirali/Projects/reference_seq.csv'

writefile = open('sequence_pam_gene_grna_big_file_donor_genomic_context_hdr.csv','w')

refrence_dic = {}
XXXX = []
with open(genomic_refrence_sheet, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    row_counter = 0
    for row in spamreader:
        if row_counter>0 and row_counter<2185:
            row_list = row[0].split(',')
            contexttt =   row_list[2].strip('"')
            contexttt = contexttt.replace('a','A')
            contexttt = contexttt.replace('t','T')
            contexttt = contexttt.replace('c','C')
            contexttt = contexttt.replace('g','G')
            refrence_dic[row_list[1].strip('"')] = contexttt
            XXXX.append(row_list[1].strip('"'))
        row_counter += 1



book = openpyxl.load_workbook(helper_sheet)
sheet = book.active
all_exps=[]
all_locations = []
for row in range(2, 8449):
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
        guide,well= str(sheet['E%d'% (all_exps.index(a)+2)].value).split('_')
        doner_id = str(sheet['H%s'% (all_exps.index(a)+2)].value)

        #print doner_id
        #print "guide", guide
        #print "well", well
        workbook = xlrd.open_workbook('/Users/amirali/Projects/CRISPR-data/Plate Maps/cr%d.xls' %int(guide))
        worksheet = workbook.sheet_by_name('Cherry-pick crRNA Library Plate')
        #print worksheet.cell(0, 0).value
        well_list = worksheet.col_values(1)
        spacer_list = worksheet.col_values(5)
        pam_list = worksheet.col_values(6)
        location_list = worksheet.col_values(7)
        #all_locations.append(location_list[well_list.index(well)][5:])
        #print a,curr_gene_name,spacer_list[well_list.index(well)],pam_list[well_list.index(well)]
        writefile.write('%s,%s,%s,%s,%s,%s,%s\n' %(a,curr_gene_name,spacer_list[well_list.index(well)],pam_list[well_list.index(well)],location_list[well_list.index(well)][5:],doner_id,refrence_dic[guide + '_' + well]))


#print "number of sites", np.shape(all_locations)
#print "number of unique sites", len(set(all_locations))