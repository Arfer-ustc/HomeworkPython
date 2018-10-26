from math import sin, cos, radians

class Projectile:
    """Simulates the flight of simple projectiles near the earth's
    surface , ignoring wind resistance . Tracking is done in two
    dimensions , height (y) and distance (x) .
    """
    def __init__(self, angle, velocity, height):
        self.posX = 0.0
        self.posY = height
        theta = radians(angle)
        self.velX = velocity * cos(theta)
        self.velY = velocity * sin(theta)
        self.maxHeight=0.0

    def getX(self):
        return self.posX

    def getY(self):
        return self.posY

    def update(self, time):
        self.posX = self.posX + self.velX * time
        nextVelY = self.velY - 9.8 * time
        self.posY = self.posY + (self.velY + nextVelY) / 2 * time
        if self.maxHeight < self.posY:
            self.maxHeight = self.posY
        self.velY = nextVelY

    def getMaxHeight(self):
        return self.maxHeight
def getInputs():
    a=float(input("Enter the launch angle(in degrees):"))
    v=float(input("Enter tne initial velocity(in meters/sec):"))
    h=float(input("Enter the initial height(in meters):"))
    t=float(input("Enter the time interval between position calculations:"))
    return a,v,h,t

def main():
    angle, vel, h0, time = getInputs()
    cball = Projectile(angle, vel, h0)
    while cball.getY() >= 0:
        cball.update(time)
    print("\nDistance traveled: {0: 0.if} meters.".format(cball.getX()))