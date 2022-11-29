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