import pygame


def change_image_color(image, color):
    colored_image = pygame.Surface(image.get_size())
    colored_image.fill(color)

    new_image = image.copy()
    new_image.blit(colored_image, (0, 0), special_flags=pygame.BLEND_MULT)
    return new_image