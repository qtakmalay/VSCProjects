# Using the built-in range(start, stop, step), write a program that reads these three values (you
# can assume correct input) and iterates through the value range. Your program must do the following:
# • Create two variables: odd_counter and even_sum (both initialized with 0).
# • If the value is odd, increment odd_counter by 1.
# • If the value is even, add it to even_sum.
# • If the value is the second value in the range iteration (if there even is such a second value),
# print 2nd value in range = X, where X is this second value.
# • If the value is the last value in the range iteration (if there even is such a last value), print
# Last value in range = X, where X is this last value.
# • After the iteration, print both the counter odd_counter as well as the sum even_sum.
# Example input:
# Start: 2
# Stop: 20
# Step: 3
# Example output:
# 2nd value in range = 5
# Last value in range = 17
# odd_counter = 3
# even_sum = 24
# Hints:
# • You can get the number of range elements via len(range_object).
start_in, stop_in, step_in, odd_counter, even_sum = int(input("Start: ")), int(input("Stop: ")), int(input("Step: ")), 0, 0
list_nums = list(range(start_in, stop_in, step_in))
for val in list_nums:
    if(val % 2 == 1):
        odd_counter += 1    
    elif(val % 2 == 0):
        even_sum += val
print(list_nums)
print(f"""2nd value in range = {list_nums[1]}
Last value in range = {list_nums[len(list_nums)-1]}
odd_counter = {odd_counter}
even_sum = {even_sum}""")
