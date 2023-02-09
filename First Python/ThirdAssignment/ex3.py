
# Enter string: This 12 IS an ExamPLE SENTEnce.
# Example output (order might differ):
# {'t': 2, 'h': 1, 'i': 2, 's': 3, ' ': 5, '1': 1, '2': 1, 'a': 2, 'n': 3,
# 'e': 5, 'x': 1, 'm': 1, 'p': 1, 'l': 1, 'c': 1, '.': 1}
import ex4 as methods
letter_dict = dict()
usr_in = input("Enter string: ").lower()
for element in usr_in:
    if(not element in letter_dict):
        letter_dict[element] = sum(1 for b in usr_in if b in element)  
print(letter_dict)



