from gameObject import GameObject

class Obstacle(GameObject):
  def __init__(self, position, size):
    image = 'img/wall.png'
    size = (size, size)

    super().__init__(image, position, size)
