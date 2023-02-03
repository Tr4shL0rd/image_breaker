import random
from rich import print
from PIL import Image


def noise(image:Image):#noise(pixels:list):
    """adds pseudo noise to an image"""
    print("[green][*][/green] Adding noise")
    pixels = list(image.getdata())
    _new_pixels = []
    for pixel in pixels:
        pixel_brightness_rng = random.randint(1,2) # rng for adding or subtracting pixel brightness
        manipulate_pixel_rng = random.randint(1,50)
        if manipulate_pixel_rng < 10:
            broken_pixel:tuple = (
                                    pixel[0] - random.randint(0,100) if pixel_brightness_rng == 1 else pixel[0] + random.randint(0,100),
                                    pixel[1] - random.randint(0,100) if pixel_brightness_rng == 1 else pixel[1] + random.randint(0,100),
                                    pixel[2] - random.randint(0,100) if pixel_brightness_rng == 1 else pixel[2] + random.randint(0,100)
                                )
        else:
            broken_pixel = (
                                pixel[0],
                                pixel[1],
                                pixel[2]
                            )
        _new_pixels.append(broken_pixel)
    print("[green][*][/green] Done adding noise")
    return _new_pixels

def horizontal_shift(image:Image):
    """shifts pixels horizontally"""
    print("[green][*][/green] Shifting pixels laterally")
    width, height = image.size
    pixels = list(image.getdata())
    _new_pixels = []
    for y in range(height):
        manipulate_pixel_chance_rng = random.randint(1,100)
        shift_distance_rng   = random.randint(0,width)
        for x in range(width):
            if manipulate_pixel_chance_rng < 10:
                new_x = (x+shift_distance_rng) % width
                new_y = (y+shift_distance_rng) % height
                _new_pixels.append(pixels[new_y * width + new_x])
            else:
                _new_pixels.append(pixels[y * width + x])
    print("[green][*][/green] Done shifting pixels")
    return _new_pixels


def combine_pixels(*pixel_lists):
    """Combines lists of pixels into one list"""
    print("[green][*][/green] Combining pixels")
    combined_pixels = []
    for pixels in zip(*pixel_lists):
        combined_pixel = [0, 0, 0]
        for pixel in pixels:
            if isinstance(pixel, int):
                pixel = (pixel, pixel, pixel)
            combined_pixel = [sum(x) for x in zip(combined_pixel, pixel)]
        combined_pixel = tuple(x//len(pixel_lists) for x in combined_pixel)
        combined_pixels.append(combined_pixel)
    return combined_pixels




IMAGE_FILE = "image.jpg"
im = Image.open(IMAGE_FILE)
noise_pixels = noise(im)
lateral_shift_pixels = horizontal_shift(im)
pixels = combine_pixels(
                            lateral_shift_pixels,
                            noise_pixels,
                        )
im.putdata(pixels)
im.save("new_image.jpg")
