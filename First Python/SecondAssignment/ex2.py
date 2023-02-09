# Write a program that counts uppercase characters in a user-specified string and prints the result
usr_input = input("Enter text: ")
print("The input text contains %s uppercase characters." %(sum(1 for b in usr_input if b.isupper())))



