def round_(number, ndigits: int = None):
    if ndigits == None:
        return int("%0.0f" % number)
    if ndigits >= 0: 
        cnst_str = "%." + str(ndigits) + "f"
        return float(cnst_str % (number))
    if ndigits < 0:
        return int(number * pow(10,ndigits) + 0.5) / pow(10,ndigits)
        
    
