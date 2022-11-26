# Write a program that calculates the ticket price of some public transport based on the age and,
# conditionally, the salary of the users. Your program should behave as follows:
#   Read the age as integer value (you can assume correct user input)
#   If the age is negative, print the error message Invalid age
#   If the age is 7 or below, print Child ticket: 10$
#   If the age is between 8 and 17 (inclusive), print Teenager ticket: 15$
#   If the age is 18 or more, the ticket price depends on the salary:
#   Read the salary as integer value (you can assume correct user input)
#   If the salary is negative, print the error message Invalid salary
#   If the salary is 1000 or below, print Reduced adult ticket 1: 20$
#   If the salary between 1001 and 2000 (inclusive), print Reduced adult ticket 2: 25$
#   In all other cases, print Adult ticket: 30$
#
# Enter age: 31
# Enter salary: 1700
# Reduced adult ticket 2: 25$


#   Read the age as integer value (you can assume correct user input)
age_inp= input("Enter age: ")
tikts_vals = (10, 15, 20, 25, 30)

#   If the age is negative, print the error message Invalid age
while(not isinstance(age_inp, int) | age_inp < 0):
    print("Ivalid age.")
    age_inp = input("Enter age: ")

#   If the age is 7 or below, print Child ticket: 10$
if(age_inp <= 7):
    print("Child ticket: %s$" %(tikts_vals[0]))
#   If the age is between 8 and 17 (inclusive), print Teenager ticket: 15$
elif(age_inp >= 8) and (age_inp <= 17): 
    print("Teenager ticket: %s$" %(tikts_vals[1]))
#   If the age is 18 or more, the ticket price depends on the salary:
elif(age_inp >= 18):
#   Read the salary as integer value (you can assume correct user input)
    salary_inp = input("Enter salary: ")
    
#   If the salary is negative, print the error message Invalid salary
    while(not isinstance(age_inp, int)) | (salary_inp < 0):
        print("Ivalid salary.")
        salary_inp = input("Enter age: ")


#   If the salary is 1000 or below, print Reduced adult ticket 1: 20$
    if(salary_inp <= 1000):
        print("Reduced adult ticket 1: %s$" %(tikts_vals[2]))
#   If the salary between 1001 and 2000 (inclusive), print Reduced adult ticket 2: 25$
    elif(salary_inp >= 1001) and (salary_inp <= 2000):
        print("Reduced adult ticket 2: %s$" %(tikts_vals[3]))
#   In all other cases, print Adult ticket: 30$
    elif(salary_inp>2000):
         print("Adult ticket: %s$" %(tikts_vals[4]))