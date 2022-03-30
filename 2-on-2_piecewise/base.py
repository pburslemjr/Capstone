from gameObject import GameObject

class Base(GameObject):
  def __init__(self, color, position, size):
    image = 'img/'+color+'_basetop.png'
    size = (size, size)

    # specific to Base
    self.color = color

    super().__init__(image, position, size)
