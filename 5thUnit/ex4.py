def flatten(nested : list):
    if not nested:
        return nested
    if isinstance(nested[0], list):
        return flatten(*nested[:1]) + flatten(nested[1:])
    return nested[:1] + flatten(nested[1:])