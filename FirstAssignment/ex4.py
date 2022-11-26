import math
a = float(input("Edge length: "))

print("Surface: %s\n Volume: %s\n Height: %s\n" %(round(a**2 * math.sqrt(3) , 4), round((a**3/12) * math.sqrt(2), 4), round((a/3) * math.sqrt(6), 4)))