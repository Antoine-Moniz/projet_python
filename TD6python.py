from turtle import *


speed(0)
hideturtle()

def virage (longueur):
    
    forward(longueur)
    right(90)
    forward(longueur)
    right(90)


def motif(longueur,repetitions):
    color("blue")
    forward(longueur)
    left(90)
   
    color("blue")
    n=0   
    while n<repetitions: 
        virage(longueur)
        longueur=longueur*2/3
        n=n+1
  
def deplaceCurseur(x,y):
    up()
    goto(x,y)
    down()  

deplaceCurseur(-300, 0)       
i=0
while i<7 :
    motif(60,4)
    x,y = pos()
    y=0
    deplaceCurseur(x, y)
    right(90)
    i=i+1   
    
