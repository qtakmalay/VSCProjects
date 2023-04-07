numbers = [1, 2, 0, 2, 0, 0, 1, 2, 0, 0, 2, 1, 2, 2, 2, 1, 1, 1, 1, 6, 0, 2, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 3, 3]


# Create a dictionary to store the counts for each unique number
count_dict = {}

# Loop through the array and update the counts in the dictionary
for num in numbers:
    if num in count_dict:
        count_dict[num] += 1
    else:
        count_dict[num] = 1

# Print out the counts for each unique number
for num, count in count_dict.items():
    print(f"{num} occurs {count} times")
    print(f"{num} rel fre {count/len(numbers)}")