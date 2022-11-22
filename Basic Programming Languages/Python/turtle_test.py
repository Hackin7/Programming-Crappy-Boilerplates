import turtle
import time
# https://realpython.com/beginners-guide-python-turtle/

#s = turtle.getscreen()
t = turtle.Turtle()

### Properties #######################
#turtle.bgcolor("blue")
turtle.title("My Turtle Program")

### Movement #########################
'''
t.right(90) # Rotate Right
t.forward(10)
t.left(90) # Rotate Left
t.backward(10)
'''
t.rt(90) # Rotate
t.fd(100)
t.lt(90) # Rotate Right
t.bk(100)
time.sleep(1)

### Drawing Preset Figures ###########
t.circle(60) # Radius
t.dot(20)
time.sleep(1)

### Turtle Properties ################
t.shapesize(3,3,3)
t.fillcolor("red")
t.pencolor("red")
t.pensize(3)
#t.pen(pencolor="purple", fillcolor="orange", pensize=10, speed=9)

t.shape("turtle")
t.shape("circle")
t.shape("arrow")


### Turtle Fill ######################
t.clear()
t.begin_fill()
t.fd(100)
t.lt(120)
t.fd(100)
t.lt(120)
t.fd(100)
t.end_fill()
time.sleep(1)

### Turtle pen up & down ##############
t.clear()
t.stamp() # Stamp of turtle
t.penup()
t.fd(100)
t.pendown()
t.fd(100)

time.sleep(1)

### Cloning Turtle ####################
t.speed(0)
t.reset()
c = t.clone()

c.shapesize(1,1,1)
t.color("magenta")
c.color("red")
t.circle(100)
c.circle(60)
t.write("GeeksForGeeks")


t.goto(100,100)
t.goto(-90,100)
time.sleep(1)
### Environment Properties ############
t.clear()
t.reset()
