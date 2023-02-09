# a b c d 
# e f g h
# i j k l
# m n o p

#Every node has a number
def rule_each_row_has_one_number_row(matrix): #input main list
    sums_matrix = str()
    for i, vals in enumerate(matrix):
        for i2, vals2 in enumerate(vals):
             sums_matrix += '('
             sums_matrix += vals2+'1 | ' + vals2+'2 | ' + vals2+'3 | ' + vals2+'4'
             sums_matrix += ') &'
    return sums_matrix
#Every node has at most one number
def rule_each_row_has_most_one_number_row(matrix): #input main list
    sums_matrix = str()
    for i, vals in enumerate(matrix):
        for i2, vals2 in enumerate(vals):
             sums_matrix += '('
             sums_matrix += '!'+vals2+'1 | ' + '!'+vals2+'2 | ' + '!'+vals2+'3 | ' + '!'+vals2+'4'
             sums_matrix += ') &'
    return sums_matrix
def rule_each_row_has_most_one_number_column(matrix): #input main list
    sums_matrix = str()
    for i, vals in enumerate(matrix):
        for i2, vals2 in enumerate(vals):
             sums_matrix += '('
             sums_matrix += '!'+vals2
             sums_matrix += ') &'
    return sums_matrix
grid = list()
grid.append(['a', 'b', 'c', 'd'])
grid.append(['e', 'f', 'g', 'h'])
grid.append(['i', 'j', 'k', 'l'])
grid.append(['m', 'n', 'o', 'p'])
print(rule_each_row_has_one_number(grid))
print(rule_each_row_has_most_one_number(grid))
