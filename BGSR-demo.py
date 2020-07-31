import numpy as np
from simulate_Data_LR_HR import simulate_Data_LR_HR
from atlas import atlas
import xlsxwriter

mu1 = 0.8 # Mean parameter of the first Gaussian distribution
sigma1 = 0.4 # Standard deviation parameter of the first Gaussian distribution

mu2 = 0.7 # Mean parameter of the second Gaussian distribution
sigma2 = 0.6 # Standard deviation parameter of the second Gaussian distribution

kn = 10 # Number of selected features

def Leave_one_out_cross_validation(LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, test_index, index):

    test_label = LR_data_av_Labels[test_index]
    test_data = LR_data_av_x[test_index][:][:]
    index = np.delete(index, test_index)
    train_labels = LR_data_av_Labels[index]
    train_data = LR_data_av_x[index][:][:]
    index = np.insert(index, test_index, test_index)

    return train_data, train_labels, test_data, test_label

LR_data_max_Featurematrix, LR_data_max_x, LR_data_max_Labels, LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, HR_data_Featurematrix, HR_data_x, HR_data_Labels = simulate_Data_LR_HR(mu1, sigma1, mu2, sigma2)


# #HR FEATURE MATRIX
# workbook = xlsxwriter.Workbook('HR_data_Featurematrix.xlsx')
# worksheet = workbook.add_worksheet()
#
# col=0
# for row, data in enumerate(HR_data_Featurematrix):
#     worksheet.write_row(row, col, data)
# workbook.close()
#
# #HR DATA
# for i in range(0, len(HR_data_x)):
#
#     workbook = xlsxwriter.Workbook('HR_data_x_{0}.xlsx'.format(i))
#     worksheet = workbook.add_worksheet()
#
#     col=0
#     for row, data in enumerate(HR_data_x[i]):
#         worksheet.write_row(row, col, data)
#     workbook.close()
#
# #HR LABELS
# workbook = xlsxwriter.Workbook('HR_data_Labels.xlsx')
# worksheet = workbook.add_worksheet()
#
# col=0
# for row, data in enumerate(HR_data_Labels):
#     worksheet.write_row(row, col, data)
# workbook.close()
#
#
#
#
# #LR MAX FEATURE MATRIX
# workbook = xlsxwriter.Workbook('LR_data_max_Featurematrix.xlsx')
# worksheet = workbook.add_worksheet()
#
# col=0
# for row, data in enumerate(LR_data_max_Featurematrix):
#     worksheet.write_row(row, col, data)
# workbook.close()
#
# #LR MAX DATA
# for i in range(0, len(LR_data_max_x)):
#
#     workbook = xlsxwriter.Workbook('LR_data_max_x_{0}.xlsx'.format(i))
#     worksheet = workbook.add_worksheet()
#
#     col=0
#     for row, data in enumerate(LR_data_max_x[i]):
#         worksheet.write_row(row, col, data)
#     workbook.close()
#
# #LR MAX LABELS
# workbook = xlsxwriter.Workbook('LR_data_max_Labels.xlsx')
# worksheet = workbook.add_worksheet()
#
# col=0
# for row, data in enumerate(LR_data_max_Labels):
#     worksheet.write_row(row, col, data)
# workbook.close()
#
#
#
# #LR AV FEATURE MATRIX
# workbook = xlsxwriter.Workbook('LR_average_Featurematrix.xlsx')
# worksheet = workbook.add_worksheet()
#
# col=0
# for row, data in enumerate(LR_average_Featurematrix):
#     worksheet.write_row(row, col, data)
# workbook.close()
#
# #LR AV DATA
# for i in range(0, len(LR_data_av_x)):
#
#     workbook = xlsxwriter.Workbook('LR_data_av_x_{0}.xlsx'.format(i))
#     worksheet = workbook.add_worksheet()
#
#     col=0
#     for row, data in enumerate(LR_data_av_x[i]):
#         worksheet.write_row(row, col, data)
#     workbook.close()
#
# #LR AV LABELS
# workbook = xlsxwriter.Workbook('LR_data_av_Labels.xlsx')
# worksheet = workbook.add_worksheet()
#
# col=0
# for row, data in enumerate(LR_data_av_Labels):
#     worksheet.write_row(row, col, data)
# workbook.close()


index = np.arange(0, len(LR_data_av_Labels))
for test_index in range(0, len(LR_data_av_Labels)):

    train_data, train_labels, test_data, test_label = Leave_one_out_cross_validation(LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, test_index, index)
    CBT = atlas(train_data, train_labels)
