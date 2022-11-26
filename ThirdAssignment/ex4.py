
def parse_list(in_string): # input string with values and spaces
    var = ''
    new_list = list()
    for i,val in enumerate(in_string):
        if not val.isspace():
            var += val
        if val.isspace() or len(in_string)-1 == i:
            new_list.append(int(var))
            var = ''
    return new_list

def max_matrix_length(matrix): # input main list
    max_length_l = 0
    for val in matrix:
        if len(val) > max_length_l:
            max_length_l = len(val)
    return max_length_l

def expand_matrix(matrix): #input main list
    max_el = max_matrix_length(matrix)
    for vals in matrix:
        while(len(vals) != max_el):
            vals.append(0)
    return matrix

def sum_matrix_rows(matrix): #input main list
    sums_matrix = list()
    for vals in matrix:
        sum_vals = 0
        for inner_vals in vals:
            sum_vals += inner_vals
        sums_matrix.append(sum_vals)
    return sums_matrix

def sum_matrix_columns(matrix): #input main list
    sums_matrix = [0] * max_matrix_length(matrix)
    for i, vals in enumerate(matrix):
        for i2, vals2 in enumerate(vals):
            sums_matrix[i2] += vals2 
    return list(sums_matrix)

def matrix_beautifier(matrix : list): #input main list
    beauty_str = "["
    for val in matrix:
        beauty_str += "["
        for val1 in val:
            beauty_str += str(val1) + " "
        if val == matrix[-1]:
            beauty_str += "]] "
        else:
            beauty_str += "]\n "
    return beauty_str 

outer_list = list()
switch = True
while switch:
    row_in = input("Enter row: ")
    if(not "x" in row_in):
        outer_list.append(parse_list(row_in)) 
    else:
        switch = False

outer_list = expand_matrix(outer_list)
print("""%s
row sums: %s
column sums: %s
total sum: %s""" %(matrix_beautifier(outer_list), sum_matrix_rows(outer_list), sum_matrix_columns(outer_list), sum(b for b in sum_matrix_rows(outer_list))))


