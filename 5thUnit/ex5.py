def f(x: int):
    try:
        g(x)
        print("f1")
    except TypeError:
        print("f2")
        raise print(KeyError)
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
            print(KeyError)
            #raise print(KeyError)
        print("g3")
    finally:
        print("g4")
def h(x: int):
    try:
        if x < 0:
            print(TypeError)
        if x > 10:
            print(ValueError)
    finally:
        print("h1")
        print("h2")
print(f(1))
print("----------------------")
print(f(-1))
print("----------------------")
print(f(15))
print("----------------------")
print(f(-15))
