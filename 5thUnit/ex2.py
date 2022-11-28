def gen_range(start: int, stop: int, step: int = 1):
    try:
        
        if step == 0:
            raise ValueError("Value Error")
        elif isinstance(start, float) or isinstance(stop, float):
            raise TypeError("Type Error")
        else:
            if step > 0:
                while start < stop:
                    yield start
                    start += step 
            if step < 0:
                while int(start) > int(stop):
                    yield start
                    start += step 
 
    except (TypeError, ValueError) as ex:
        print(f"We caught the exception '{ex}'")





print(list(gen_range(0, 10))) #= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(list(gen_range(0, 10, 3))) #= [0, 3, 6, 9]
print(list(gen_range(0, 10, -1))) #= []
print(list(gen_range(10, 0))) #= []
print(list(gen_range(10, 0, -2))) #= [10, 8, 6, 4, 2]
print(list(gen_range(-10, -3, 2))) #= [-10, -8, -6, -4]
print(list(gen_range(0.0, 10))) #-> TypeError
print(list(gen_range(0, 10, 0))) #-> ValueError