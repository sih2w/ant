import pygame

def fill(surface, color):
    w, h = surface.get_size()
    r, g, b = color
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            surface.set_at((x, y), pygame.Color(r, g, b, a))

class LayeredSprite:
    def __init__(
            self,
            foreground_image: str,
            background_image: str or None = None,
            dimensions: (int, int) = (100, 100),
            rotation: float = 0,
            color: (int, int, int) or None = None,
    ):
        self.__foreground_image = pygame.image.load(foreground_image)
        self.__foreground_image = pygame.transform.scale(self.__foreground_image, dimensions)
        if rotation != 0:
            self.__foreground_image = pygame.transform.rotate(self.__foreground_image, rotation)
        fill(self.__foreground_image, color)

        self.__background_image = pygame.image.load(background_image)
        self.__background_image = pygame.transform.scale(self.__background_image, dimensions)
        if rotation != 0:
            self.__background_image = pygame.transform.rotate(self.__background_image, rotation)

    def draw(self, canvas, position):
        canvas.blit(self.__background_image, position)
        canvas.blit(self.__foreground_image, position)