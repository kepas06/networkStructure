
from random import randrange
from collections import deque,Counter
import math
import numpy as np


def linear(a,b):
        return a*b

def sigmoid(x):
    return (1)/(1 + math.exp(-x))

def prog(x):
    initial = 0
    if x >=0:
        initial = 1
    else:
        initial = 0
    return initial



#pierwsza siec 
class oneNeuron:
    def __init__(self, x,x1,weights):
        self.input1 = x
        self.input2 = x1
        self.w1= weights#np.random.randint(-2,2,size=(1,2))

    def wagi(self):     
        suma = self.w1[0][0]*self.input1+self.w1[0][1]*self.input2
        return suma #zwraca tab [1 2]              

    def linear_activ(self,a):        
        suma = self.wagi()
        return  linear(a,suma)

    def sigmoid_activ(self):    
        suma = self.wagi()
        return  sigmoid(suma)

    def prog_activ(self):  
        suma = self.wagi()
        y = prog(suma)
        return y
# ####################################################################################################################################
#ze stalym sygnalem 


class oneNeuronBias:
    def __init__(self, x,x1,weights,signal):
        self.input1 = x
        self.input2 = x1
        self.w1= weights#np.random.randint(-2,2,size=(1,2))      
        self.bias=signal

    def wagiBias(self):
        suma = self.w1[0][0]*self.input1+self.w1[0][1]*self.input2+self.w1[0][2]*self.bias
        return suma #zwraca tab [1 2]        

    def linear_activ(self,a):        
        suma = self.wagiBias()
        result = linear(a,suma)
        return  result

    def sigmoid_activ(self):    
        suma = self.wagiBias()  
        return  sigmoid(suma)

    def prog_activ(self):  
        suma = self.wagiBias()
        y = prog(suma)
        return y

class twoNeurons:
    def __init__(self, x,x1,weights):
        self.input1 = x
        self.input2 = x1
        self.w= weights

    def wagi(self):
        o1 = ( self.w[0][0]*self.input1+ self.w[1][0]*self.input2)
        return o1 #zwraca tab [1 2]

    def wagi1(self):
        o2 = ( self.w[0][1]*self.input1+ self.w[1][1]*self.input2)
        return o2 #zwraca tab [1 2]
    
    def linear_activ(self,a):        
        suma = self.wagi()
        suma1 = self.wagi1()
        y1 = linear(a,suma)
        y2 = linear(a,suma1)
        sumaK = y1*self.w[2][0]+y2*self.w[2][1]
        y3 = linear(a,sumaK)
        return y3


    def sigmoid_activ(self):    
        suma = self.wagi()
        suma1 = self.wagi1()
        y1 = sigmoid(suma)
        y2 = sigmoid(suma1)
        sumaK = y1*self.w[2][0]+y2*self.w[2][1]
        y3 = sigmoid(sumaK)
        return y3


    def prog_activ(self):  
        suma = self.wagi()
        suma1 = self.wagi1()
        y1 = prog(suma)
        y2 = prog(suma1)
        sumaK = y1*self.w[2][0]+y2*self.w[2][1]
        y3 = prog(sumaK)
        return y3
        

class twoNeuronsBias:
    def __init__(self, x,x1,weights,signal,weightBias):
        self.input1 = x
        self.input2 = x1
        self.w= weights
        self.bias = signal
        self.wb = weightBias

    def neur1(self):
        o1 = ( self.w[0][0]*self.input1+ self.w[1][0]*self.input2)
        o2 = ( self.wb*self.bias )
        suma = o1+o2
        return suma

    def neur2(self):
        o1 = ( self.w[0][1]*self.input1+ self.w[1][1]*self.input2)
        o2 = ( self.wb*self.bias )
        suma = o1+o2
        return suma

    def linear_activ(self,a):
        o2 = ( self.wb*self.bias )        
        suma = self.neur1()
        suma1 = self.neur2()
        y1 = linear(a,suma)*self.w[2][0]
        y2 = linear(a,suma1)*self.w[2][1]
        sumaK = y1 + y2 +o2
        return linear(a,sumaK)

    def sigmoid_activ(self):
        o2 = ( self.wb*self.bias )        
        suma = self.neur1()
        suma1 = self.neur2()
        y1 = sigmoid(suma)*self.w[2][0]
        y2 = sigmoid(suma1)*self.w[2][1]
        sumaK = y1 + y2 +o2
        return sigmoid(sumaK)    
        
    def prog_activ(self):
        o2 = ( self.wb*self.bias )        
        suma = self.neur1()
        suma1 = self.neur2()
        y1 = prog(suma)*self.w[2][0]
        y2 = prog(suma1)*self.w[2][1]
        sumaK = y1 + y2 +o2
        return prog(sumaK)   
        



