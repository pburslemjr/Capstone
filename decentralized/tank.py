from gameObject import GameObject
from pygame.math import Vector2
from bullet import Bullet
import math
import gameConsts
import random

import pygame as pg
random.seed(1)

#screen = gameConsts.screen

class Tank(GameObject):
    def __init__(self, color, position, number, addToGameObject):
      # general attributes
      image = 'img/'+color+'_tank.png'
      size = gameConsts.TANK_SIZE
      direction = (1,0) # direction the image naturally faces
      speed = gameConsts.TANK_MAX_SPEED
      angle = 0

      # specific to tank
      self.color = color
      self.destination = position
      self.startingPosition = position
      self.tank_num = number
      self.selected = False
      self.font = None
      if gameConsts.render:
          self.font = pg.font.SysFont(gameConsts.TANK_FONT, gameConsts.TANK_FONT_SIZE)
          self.text = self.font.render(str(number), False, gameConsts.SELECTED_COLOR)
      self.flag = None
      self.respawn = False
      self.timeOfDeath = 0
      self.ghost = False
      self.addToGameObject = addToGameObject
      self.fired = 0
      self.reload_time = 100
      self.ghost_time = 5
      self.ghost_mode = 0

      self.speed = 0.0
      self.angleSpeed = 0.0

      super().__init__(image, position, size, direction, speed, angle)

      # start at random angle
      randomAngle = random.randint(0, 360)
      self.direction.rotate_ip(randomAngle)
      self.angle = round((self.angle + randomAngle) % 360, 2)
      if gameConsts.render:
          self.image = pg.transform.rotate(self.original_image, -self.angle)
      self.rect = self.image.get_rect(center=self.rect.center)

    def update(self, enable):
        xDiff = self.destination[0] - self.position[0]
        yDiff = self.destination[1] - self.position[1]
        if(abs(xDiff) > 1 and abs(yDiff) > 1): # if at destination, don't calc angleSpeed
            destVector = Vector2(xDiff, yDiff)
            destAngle = round(self.direction.angle_to(destVector), 2)
            if (destAngle > 180): # compensate for destAngle_to picking the wrong direction in these cases
                destAngle = -360 + destAngle
            if (destAngle < -180):
                destAngle = 360 + destAngle
            if(enable == 1):
                self.angleSpeed = round(self.getMaxRotation(destAngle), 2)

        distance = math.hypot(self.position[0] - self.destination[0], self.position[1] - self.destination[1])
        if(enable == 1):
            if distance < self.radius:
                self.speed = round(4 * distance / self.radius, 2)

            else:
                self.speed = gameConsts.TANK_MAX_SPEED
        if (self.ghost):# and not(self.preventDirection['up'] or self.preventDirection['right'] or self.preventDirection['down'] or self.preventDirection['left'])):
            if(self.ghost_mode > self.ghost_time):
                self.ghost = False
                self.ghost_mode = 0
            else:
                self.ghost_mode = self.ghost_mode + 1

        if(self.fired > 0):
            self.fired = self.fired + 1
        if(self.fired == self.reload_time):
            self.fired = 0

        super().update()
    '''if self.selected:
      #pg.draw.circle(screen, gameConsts.SELECTED_COLOR, self.position, self.radius, 1)
      pg.draw.rect(screen, gameConsts.SELECTED_COLOR, [self.position[0]-self.radius,self.position[1]-self.radius,2*self.radius,2*self.radius], 1)
    # self.text = self.font.render(f'{str(self.angle)}-{str(self.direction)}', False, gameConsts.SELECTED_COLOR) # DEBUG
    screen.blit(self.text, (self.position[0], self.position[1] + self.radius))'''



    def getMaxRotation(self, desiredAngle):
        if (desiredAngle > gameConsts.TANK_MAX_ROTATION):
          return gameConsts.TANK_MAX_ROTATION
        elif (desiredAngle < -gameConsts.TANK_MAX_ROTATION):
          return -gameConsts.TANK_MAX_ROTATION
        else:
          return desiredAngle

    def setDestination(self, pos):
        self.destination = pos

    def fire(self):
        self.fired = 1
        bulletRadius = gameConsts.BULLET_SIZE / 2 / math.cos(45)
        frontOfTank = self.position + self.direction * (self.radius + bulletRadius) # TANK_MAX_SPEED cases when tank postion is updated before bullet
        bullet = Bullet(self.color, frontOfTank, self.direction, self.angle, self.tank_num)
        self.addToGameObject(bullet)
        return bullet

    def select(self):
        self.selected = True

    def unselect(self):
        self.selected = False

    def setFlag(self, flag):
        self.flag = flag

    def setRespawn(self, timeOfDeath):
        self.respawn = True
        self.fired = 0
        self.DeathStep = timeOfDeath
        self.position = Vector2((-gameConsts.TANK_SIZE[0], -gameConsts.TANK_SIZE[1])) # off screen
        self.setDestination(self.position)
        self.updateSides()

    def checkRespawn(self, currentStep):
        if (currentStep > self.DeathStep + random.randint(100,200)):
          self.respawn = False;
          self.position = self.startingPosition
          self.setDestination(self.position)
          self.ghost = True
