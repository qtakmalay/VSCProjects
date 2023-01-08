import math

class Aggregator:
    def __init__(self, agg_type: type, ignore_errors: bool = True):
        raise NotImplementedError
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

int_agg = Aggregator(agg_type=int)
int_agg(1, 2, 3)
int_agg(4, "hi", 5.1)
print(int_agg())
str_agg = Aggregator(agg_type=str, ignore_errors=False)
print(str_agg("this", " ", "is a test"))
try:
    str_agg(1)
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
