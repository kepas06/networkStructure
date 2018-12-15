import pygame,random, sys,time

from random import randrange,uniform
from collections import deque,Counter
import math
import numpy as np
import networks as nt
import array as arr

screen = pygame.display.set_mode((400,400))
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
darkBlue = (0,0,128)
white = (255,255,255)
black = (0,0,0)
yellow =(255, 255, 0)
pink = (255,200,200)

screen.fill(black)


def linear_check(value,x,y):
    if ( value< -2):
        screen.set_at((int(x*100),int(y*100)), darkBlue)
    elif ( value>=-2 and value<0):
        screen.set_at((int(x*100),int(y*100)), blue)
    elif ( value>=0 and value<2):
        screen.set_at((int(x*100),int(y*100)), green)
    elif ( value>2):
        screen.set_at((int(x*100),int(y*100)), red)
    else:
        pass

def sigmoid_check(value,x,y):
    if ( value>= 0.0 and value<0.25):
        screen.set_at((int(x*100),int(y*100)), yellow)
    elif ( value>= 0.25 and value<0.5):
        screen.set_at((int(x*100),int(y*100)), blue)
    elif ( value>= 0.5 and value<0.75):
        screen.set_at((int(x*100),int(y*100)), green)
    elif ( value>= 0.75 and value<1.0):
        screen.set_at((int(x*100),int(y*100)), red)
    else:
        pass

def prog_check(value,x,y):
    if ( value== 1):
        screen.set_at((int(x*100),int(y*100)), red)
    elif ( value==0):
        screen.set_at((int(x*100),int(y*100)), blue)
    else:
        pass


weights = [[uniform(-2, 2),uniform(-2, 2)]]
weightsBias = [[uniform(-2, 2),uniform(-2, 2),uniform(-2, 2)]]

weights2 = [[uniform(-2,2)  for x in range(2)] for x in range(3)]


wagaBiasu = uniform(-2,2)
wagaBiasu2 = uniform(-2,2)
for a in range(400):
    for b in range(400):
        x = b/100 -2
        y = a/100 -2
        obj1 = nt.oneNeuron(x,y,weights)
        obj2 = nt.oneNeuronBias(x,y,weightsBias,wagaBiasu)
        obj3 = nt.twoNeurons(x,y,weights2)   
        obj4 = nt.twoNeuronsBias(x,y,weights2,-1,wagaBiasu)

        #1n
        value1 = obj1.linear_activ(1)
        value2 = obj1.sigmoid_activ()
        value3 = obj1.prog_activ()

        #1n + bias
        value1 = obj2.linear_activ(1)
        value2 = obj2.sigmoid_activ()
        value3 = obj2.prog_activ()

        #3n
        value1 = obj3.linear_activ(1)
        value2 = obj3.sigmoid_activ()
        value3 = obj3.prog_activ()

        #3n + bias
        value1 = obj4.linear_activ(1)
        value2 = obj4.sigmoid_activ()
        value3 = obj4.prog_activ()

        
        x+=2
        y+=2
        # linear_check(value1,x,y)
        # sigmoid_check(value2,x,y)
        prog_check(value3,x,y)


print(weights2)
while True:
    pygame.display.update()
     #Clock.tick(240)

