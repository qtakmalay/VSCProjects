ci = """(d | !e | b | a) & 
(c | !b | !e | !d) & 
(a | !b | !d | c) & 
(!c | d | !b | !a) & 
(c | b | d | !e) & 
(a | b | !d | e) & 
(!d | !b | c | !e) & 
(c | !b | !a | !e) & 
(!e | !a | b | d) & 
(b | !d | c | !a) & 
(c | !e | !b | !d) & 
(c | !a | !d | e) & 
(c | !e | a | !d) & 
(c | d | !b | a) & 
(d | b | !c | !a) & 
(d | b | !c | !e) & 
(b | !d | c | !e) & 
(d | a | e | c) &
(!d | !e | b | a | !c) &
(!d | !e | b | !a | !c) &
(!d | !e | !b | a | !c) &
(!d | !e | !b | !a | !c) &
(d | e | b | a | c) &
(d | !e | !b | a | !c) &
(!d | e | !b | a | !c) &
(d | e | !b | a | !c) &
(d | e | b | a | !c) &
(d | e | !b | !a | c) &
(d | e | b | !a | c) &
(!d | e | !b | !a | !c) &
(!d | e | b | !a | !c)
"""
print ('The ASCII value of given character is: ', ''.join(str(ord(c)) for c in ci))
