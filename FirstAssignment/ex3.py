firstNumber = input("1st number: ")
secondNumber = input("2nd number: ")
# The sum of the two numbers
def sum_func(first, second):
    return int(first) + int(second)
# The result of the first number minus the second number, i.e., the difference
def difference_func(first, second):
    return int(first) - int(second)
# The product of the two numbers
def product(first, second):
    return int(first) * int(second)
# The first number to the power of the second number
def power_func(first, second):
    return int(first) ** int(second)
# The result of an integer division when dividing the first number by the second number
def int_division_func(first, second):
    third = int(first) / int(second)
    return int(third)
# The result of a regular division when dividing the first number by the second number
def reg_division_func(first, second):
    return int(first) / int(second)
# The remainder of an integer division (modulo) when dividing the first number by the second
# number
def modulo_func(first, second):
    return int(first) % int(second)

print("Sum: %s\n Difference: %s\n Product: %s\n Power: %s\n Quotient (int): %s\n Quotient (float): %s\n Remainder: %s\n"  %(sum_func(firstNumber, secondNumber), difference_func(firstNumber, secondNumber), product(firstNumber, secondNumber), power_func(firstNumber, secondNumber), int_division_func(firstNumber, secondNumber), reg_division_func(firstNumber, secondNumber), modulo_func(firstNumber, secondNumber)))

