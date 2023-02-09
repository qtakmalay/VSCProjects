 
# outer_list = list()
# switch = True
# max_length_l = 0
# sums_matrix_rows = list()
# sum_matrix_columns = list()
# while switch:
#     row_in = input("Enter row: ")
#     if(not "x" in row_in):
#         var = ''
#         new_list = list()
#         for i,val in enumerate(row_in):
#             if not val.isspace():
#                 var += val
#             if val.isspace() or len(row_in)-1 == i:
#                 new_list.append(int(var))
#                 var = ''
#         outer_list.append(new_list) 
#         for val in outer_list:
#             if len(val) > max_length_l:
#                 max_length_l = len(val)
#         for vals in outer_list:
#             while(len(vals) != max_length_l):
#                 vals.append(0)
#         for vals in outer_list:
#             sum_vals = 0
#         for inner_vals in vals:
#             sum_vals += inner_vals
#         sums_matrix_rows.append(sum_vals)
#         sums_matrix = [0] * max_length_l
#         for i, vals in enumerate(outer_list):
#             for i2, vals2 in enumerate(vals):
#                 sums_matrix[i2] += vals2 
#         sum_matrix_columns.append(sums_matrix)
#     else:
#         switch = False
# print("""%s
# row sums: %s
# column sums: %s
# total sum: %s""" %(outer_list, sums_matrix_rows, sum_matrix_columns[-1], sum(b for b in sums_matrix_rows)))
weights = '130 130.5'
weights.split()
for val in weights:
    int(val)
print(weights.split())