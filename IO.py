import numpy as np
import Enviroment as env
import matplotlib
import matplotlib.pyplot as plt
import pickle

def save(param,arg):
    file_name = arg+'.pickle'
    file = open(file_name,'wb')
    pickle.dump(param,file)
    print(f"Saved {arg}!")
            
def load(arg):
    file_name = arg+'.pickle'
    file = open(file_name,'rb')
    res = pickle.load(file)
    return res