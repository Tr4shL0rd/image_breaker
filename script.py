"""module for manipulating image data in order for getting interesting results"""
import random
import argparse
from typing import List
from pathlib import Path
from rich import print #pylint: disable=redefined-builtin
from PIL import Image

VERSION = "0.5.3"

parser = argparse.ArgumentParser(
                                prog="dataMosh.py",
                                description="image datamoshing",
                                epilog="Made by @Tr4shL0rd"
                                )
parser.add_argument(
                    "-i","--image",
                    dest="image",
                    action="store",
                    help="image to manipulate",
                    )

args = parser.parse_args()

def noise(image:Image) -> List:
    """
    Adds pseudo noise to an image

    Parameters:
    ----------
        * image `Image`: an instance of the Image class

    Returns:
    -------
        * _new_pixels `list`: a list of pixels with added noise
    """
    print("[green][*][/green] Adding noise")
    img_pixels = list(image.getdata())
    _new_pixels = []
    for pixel in img_pixels:
        # RNG for adding or subtracting pixel brightness
        pixel_brightness_rng = random.randint(1,2)  
        # RNG to determine if a pixel should be manipulated
        manipulate_pixel_rng = random.randint(1,50) 
        if manipulate_pixel_rng < 10:
            # Subtract or add random values from the color channels to create a "broken" pixel
            broken_pixel:tuple = (
                                pixel[0] - random.randint(0,100)
                                    if pixel_brightness_rng == 1 else 
                                        pixel[0] + random.randint(0,100),
                                pixel[1] - random.randint(0,100)
                                    if pixel_brightness_rng == 1 else 
                                        pixel[1] + random.randint(0,100),
                                pixel[2] - random.randint(0,100)
                                    if pixel_brightness_rng == 1 else 
                                        pixel[2] + random.randint(0,100)
                                )
        else:
            broken_pixel = (pixel[0],pixel[1],pixel[2])
        _new_pixels.append(broken_pixel)
    #print("[green][*][/green] Done adding noise")
    return _new_pixels

def horizontal_shift(image:Image) -> List:
    """
    Shifts the pixels in an image horizontally

    Parameters:
    ----------
        * image `Image`: input image

    Returns:
    -------
        * _new_pixels `list`: a list of shifted pixels
    """
    print("[green][*][/green] Shifting pixels laterally")
    width, height = image.size
    img_pixels = list(image.getdata()) # RGB data of each pixel
    _new_pixels = []
    for _y in range(height):
        # Generate a random integer between 1 and 100
        manipulate_pixel_chance_rng = random.randint(1,100)
        # Generate a random integer between 0 and the width of the image
        shift_distance_rng = random.randint(0,width)
        for _x in range(width):
            # checks if the pixels is to be shifted
            if manipulate_pixel_chance_rng < 10:
                # Calculate the new x and y positions for the shifted pixel
                new_x = (_x + shift_distance_rng) % width
                new_y = (_y + shift_distance_rng) % height
                # Append the shifted pixel to the new_pixels list
                _new_pixels.append(img_pixels[new_y * width + new_x])
            else:
                # Append the original pixel to the new_pixels list
                _new_pixels.append(img_pixels[_y * width + _x])
    #print("[green][*][/green] Done shifting pixels")
    return _new_pixels



def combine_pixels(*pixel_lists) -> List:
    """
    Combines lists of pixels into one list

    Parameters:
    ----------
        * *pixel_lists `list`: variable number of lists of pixels

    Returns:
    -------
        * combined_pixels `list`: a list of combined pixels
    """
    print("[green][*][/green] Combining pixels")
    combined_pixels = []
    for img_pixels in zip(*pixel_lists):
        combined_pixel = [0, 0, 0]
        # Loop over the pixels and add the corresponding color channels
        for pixel in img_pixels:
            # If the pixel is an integer, convert it to a 3-tuple with identical values
            if isinstance(pixel, int):
                pixel = (pixel, pixel, pixel)
            # Add the pixel values to the combined pixel
            combined_pixel = [sum(x) for x in zip(combined_pixel, pixel)]
        # Divide the combined pixel values by the number of input lists to get the average
        combined_pixel = tuple(x//len(pixel_lists) for x in combined_pixel)
        combined_pixels.append(combined_pixel)
    return combined_pixels



def main():
    """Main entery point"""
    
    image_file = Path(args.image)#"image.jpg"
    new_image_file = f"new_{image_file}"
    im = Image.open(image_file)#pylint: disable=invalid-name

    noise_pixels = noise(im)
    horizontal_shift_pixels = horizontal_shift(im)
    pixels = combine_pixels(
                            horizontal_shift_pixels,
                            noise_pixels,
                            )
    im.putdata(pixels)

    im.save(new_image_file)
    print(f"[green][DONE][/green] Saved to {new_image_file}")

if __name__ == "__main__":
    main()
