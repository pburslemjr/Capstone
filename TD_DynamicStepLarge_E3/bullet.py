from gameObject import GameObject
import gameConsts

class Bullet(GameObject):
  def __init__(self, color, position, direction, angle, tank):
    self.color = color
    self.tank_num = tank
    image = 'img/'+color+'_tank.png'
    size = (gameConsts.BULLET_SIZE, gameConsts.BULLET_SIZE)
    speed = gameConsts.BULLET_SPEED

    super().__init__(image, position, size, direction, speed, angle)

  def update(self):
    x, y = self.position
    if(x < 0 or x > 800 or y < 0 or y > 800): # TODO: replace number with map height and width constant
      self.markedForTermination = True
    super().update()
