# Write a program where you can guess some integer number that was entered by someone (game
# master) beforehand (you can assume correct user input). The user (player) can now guess the
# number by entering values in the console (for the sake of the guessing game, you can assume that
# the player did not see the previous game master input). Run the program until the user either
# guessed the correct integer or entered the string "exit" (you can assume correct user input, i.e.,
# either integer numbers or the string "exit"). If the input is smaller than the integer, print Your
# number is smaller. If the input is bigger, print Your number is bigger. If the numbers match,
# print Congratulations!. If the user entered the string "exit", print You lost!.

# Example program execution:
# Enter number to guess: 7
# Enter number: 3
# Your number is smaller
# Enter number: 6
# Your number if smaller
# Enter number: 8
# Your number is bigger
# Enter number: exit
# You lost!
import re

mstr_usr_in, usr_in = int(input("Enter number to guess: ")), ""


while(usr_in != "exit"):
    usr_in = input("Enter number: ")
    num_check = re.compile(r'^\-?[1-9][0-9]*$')
    if(re.match(num_check,usr_in)):
        if(int(usr_in) < mstr_usr_in):
            print("Your number is smaller")
        elif(int(usr_in) > mstr_usr_in):
            print("Your number is bigger")
        elif(int(usr_in) == mstr_usr_in):
            print("Congratulations!")
            break
    elif(usr_in == "exit"):
        print("You lost!")