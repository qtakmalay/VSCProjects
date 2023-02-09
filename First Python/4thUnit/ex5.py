def sort(elements: list, ascending: bool = True):
    for i in range(0,len(elements)-1):  
        for j in range(len(elements)-1):  
            if(ascending and elements[j]>elements[j+1]) or (not ascending and elements[j]<elements[j+1]):  
                temp = elements[j]  
                elements[j] = elements[j+1]  
                elements[j+1] = temp 
    print(elements) 


