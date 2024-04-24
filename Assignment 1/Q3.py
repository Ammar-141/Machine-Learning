# 12th March 2024
# CSC354 – Assignment1 – ML – Concept Learning
# Muhammad Ammar Mukhtar
# FA21-BSE-141
# Candidate elimination algorithm is implimented on csv dataset


import numpy as np 
import pandas as pd

data = pd.read_csv('table.csv')
concepts = np.array(data.iloc[:,0:-1])
print("\nInstances:\n",concepts)
target = np.array(data.iloc[:,-1])
print("\nTarget Values: ",target)

def learn(concepts, target): 
    specific_boundary = concepts[0].copy()
    print("\nInitialization of Specific Boundary and Generic Boundary")
    print("\nSpecific Boundary: ", specific_boundary)
    generic_boundary = [["?" for _ in range(len(specific_boundary))] for _ in range(len(specific_boundary))]
    print("\nGeneric Boundary: ",generic_boundary)  

    for i, h in enumerate(concepts):
        print("\nInstance", i+1 , ": ", h)
        if target[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_boundary)): 
                if h[x]!= specific_boundary[x]:                    
                    specific_boundary[x] ='?'                     
                    generic_boundary[x][x] ='?'
                   
        if target[i] == "no":            
            print("Instance is Negative ")
            for x in range(len(specific_boundary)): 
                if h[x]!= specific_boundary[x]:                    
                    generic_boundary[x][x] = specific_boundary[x]                
                else:                    
                    generic_boundary[x][x] = '?'        
        
        print("Specific Boundary after Instance", i+1, ": ", specific_boundary)         
        print("Generic Boundary after Instance", i+1, ": ", generic_boundary)
        print("\n")

    indices = [i for i, val in enumerate(generic_boundary) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        generic_boundary.remove(['?', '?', '?', '?', '?', '?']) 
    return specific_boundary, generic_boundary

final_specific_boundary, final_generic_boundary = learn(concepts, target)

print("Final Specific Boundary: ", final_specific_boundary, sep="\n")
print("Final Generic Boundary: ", final_generic_boundary, sep="\n")
