def f(x: int):
    try:
        g(x)
        print("f1")
    except TypeError:
        print("f2")
        print("ErrorC") 
        raise KeyError 
    except ValueError:
        print("f3")
    else:
        print("f4")
    print("f5")
def g(x: int):
    try:
        h(x)
        print("g1")
    except TypeError:
        print("g2")
        if x < -10:
            print("ErrorC") 
            raise KeyError 
        print("g3")
    finally:
        print("g4")
def h(x: int):
    try:
        if x < 0:
            print("ErrorA") 
            raise TypeError
        if x > 10:
            print("ErrorB") 
            raise ValueError
    finally:
        print("h1")
print("h2")


f(1)
print("--------")
f(-1)
print("--------")
f(15)
print("--------")
# f(-15)