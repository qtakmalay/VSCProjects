# Write a program that reads a string as input. This string is the processed character by character.
# Every lowercase character is added to a list, every uppercase letter is added to another list and
# all remaining/other characters are also added to yet another list. In addition, collect all unique
# characters in a set. Print all data structures afterwardsExample input:
# Enter string: This 12 IS an ExamPLE SENTEnce.
# Example output (order of unique might differ):
# lowercase: ['h', 'i', 's', 'a', 'n', 'x', 'a', 'm', 'n', 'c', 'e']
# uppercase: ['T', 'I', 'S', 'E', 'P', 'L', 'E', 'S', 'E', 'N', 'T', 'E']
# other: [' ', '1', '2', ' ', ' ', ' ', ' ', '.']
# unique: {'e', 'n', 'i', 'N', '2', 'c', 'E', 'h', 'a', 'I', 'x', '.', 'S',
# 's', 'm', 'L', 'T', ' ', '1', 'P'}
lower, upper, other, unique = list(), list(), list(), list()
user_in = input("Enter string:")
for element in user_in:
    if(element.isupper()):
        lower.append(element)
    if(element.islower()):
        upper.append(element)
    if(not element.isupper() and not element.islower()):
        other.append(element)
    if(not element in unique):
        unique.append(element)

print("""lowercase: %s
uppercase: %s
other: %s
unique: %s""" %(lower, upper, other, unique))