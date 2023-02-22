"""module for manipulating image data in order for getting interesting results"""
#!/usr/bin/env python3

import random
import argparse
import os.path
from datetime import datetime
from typing import List
from pathlib import Path
from tqdm import tqdm #pylint: disable=import-error
from rich import print #pylint: disable=redefined-builtin, import-error
from PIL import Image #pylint: disable=import-error

VERSION = "0.6.0"

VERBOSE_STRING = "[yellow][VERBOSE][/yellow]"

CORRECT = "[green][*][/green]"
DONE    = "[green][DONE][/green]"
NOTICE  = "[yellow][!][/yellow]"

#TODO: maybe edit metadata to show what manipulations has been done to the image
#TODO: allow multiple files to be given from command line and loop over each file

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
parser.add_argument(
                    "-v","--verbose",
                    dest="verbose",
                    action="store_true",
                    help="verbose output"
                    )
parser.add_argument(
                    "-d","--default-output",
                    dest="default_path",
                    action="store_true",
                    help="force output to the default location"
                    )
parser.add_argument(
                    "--no-progress",
                    dest="no_progress",
                    action="store_true",
                    help="disable progress bars"
                    )
parser.add_argument(
                    "-q","--quiet",
                    dest="quiet",
                    action="store_true",
                    help="quiet output"
                    )
parser.add_argument(
                    "-o","--output",
                    dest="output_name",
                    action="store",
                    help="name of the file outputted"
                    )
args = parser.parse_args()

def current_time(just_time=False,just_date=False) -> str:
    """
    returns current time

    PARAMS:
    -------
        * just_time `bool`: return only the time
        * just_date `bool`: return only the date

    RETURNS:
    -------
        * returns the date and or time
    """
    if just_time:
        return datetime.now().strftime("[%H:%M:%S]")
    if just_date:
        return datetime.now().strftime("[%d/%m/%Y]")
    if just_date and just_time :
        return datetime.now().strftime("[%d/%m/%Y|%H:%M:%S]")
    else:
        return datetime.now().strftime("[%d/%m/%Y|%H:%M:%S]")

def noise(image:Image) -> List:
    """
    Adds pseudo noise to an image

    PARAMS:
    -------
        * image `Image`: an instance of the Image class

    RETURNS:
    --------
        * _new_pixels `list`: a list of pixels with added noise
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} ADDING NOISE TO {image.filename} [NOISE()]")
        else:
            print(f"{CORRECT} Adding noise")
    img_pixels = list(image.getdata())
    _new_pixels = []
    for pixel in tqdm(
                    img_pixels,
                    total=len(img_pixels),
                    desc="Adding noise",
                    bar_format="{desc}: {percentage:3.0f}% |{bar}|",
                    leave=False,
                    disable=args.no_progress or args.quiet):
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
    return _new_pixels

def shift(image:Image) -> List:
    """
    Shifts the pixels of an image

    PARAMS:
    -------
        * image `Image`: input image

    RETURNS:
    --------
        * _new_pixels `list`: a list of shifted pixels
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "\
                    f"ADDING PIXEL SHIFTING TO {image.filename} [SHIFT()]")
        else:
            print(f"{CORRECT} Shifting pixels")
    width, height = image.size
    img_pixels = list(image.getdata()) # RGB data of each pixel
    _new_pixels = []
    for _y in tqdm(
                    range(height),
                    total=height,
                    desc="shifting",
                    bar_format="{desc}: {percentage:3.0f}% |{bar}|",
                    leave=False, # removes the progress bar after finishing
                    # Disables the progress bar if quiet
                    disable=args.no_progress or args.quiet):
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
    return _new_pixels

def combine_pixels(*pixel_lists) -> List:
    """
    Combines lists of pixels into one list

    PARAMS:
    -------
        * *pixel_lists `list`: variable number of lists of pixels

    RETURNS:
    --------
        * combined_pixels `list`: a list of combined pixels
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} COMBINING {len(pixel_lists)} "\
                                                "LISTS OF PIXELS [COMBINE_PIXELS()]")
        else:
            print(f"{CORRECT} Combining pixels")
    combined_pixels = []
    for img_pixels in tqdm(
                            zip(*pixel_lists),
                            desc="combining pixels",
                            total=len(pixel_lists[0]),
                            bar_format="{desc}:{percentage:3.0f}% |{bar}|",
                            leave=False,
                            disable=args.no_progress or args.quiet):
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
    if not args.quiet:
        print(f"{current_time()} Starting")

    # path to image (image file included)
    image_file = Path(args.image).absolute()
    # path to folder containing image file
    image_path = Path(image_file.parent)
    file_name = Path(f"new_{image_file.name}")
    if not image_file.exists():
        print(f"[red underline][WARNING] FILE \"{image_file}\" WAS NOT FOUND[/red underline]")
        exit()
    if args.output_name:
        file_name = Path(args.output_name)
        if not file_name.suffix:
            print(f"[red underline][WARNING] NO FILE EXTENSION GIVEN! "\
                f"using \"{image_file.suffix}\"[/red underline]")
            file_name = f"{file_name}{image_file.suffix}"

    # path to new image (image file included)
    default_path = Path(os.path.join(Path.cwd(), "output", file_name))
    # path to output folder
    default_output_path = Path(os.path.join(Path.cwd(), "output"))
    # creates a new default output folder if it isnt already there
    if not default_output_path.exists():
        print(f"{NOTICE} Creating default output folder!")
        os.makedirs(default_output_path)
    # checks if working path is the same as image_path
    if image_path == Path.cwd() or args.default_path:
        new_image_file = default_path
        if args.verbose and not args.quiet:
            print(f"{current_time()}{VERBOSE_STRING} USING DEFAULT OUTPUT LOCATION: "\
            f"{new_image_file} [MAIN()]")
    else:
        # deciding if output should be a working dir or image dir
        #new_image_file = Path(os.path.join(image_path,f"new_{image_file.name}"))
        new_image_file = Path(os.path.join(Path.cwd(), "output",f"new_{image_file.name}"))
    im = Image.open(image_file)#pylint: disable=invalid-name

    noise_pixels = noise(im)
    shift_pixels = shift(im)
    pixels = combine_pixels(
                            shift_pixels,
                            noise_pixels,
                            )
    if args.verbose and not args.quiet:
        print(f"{current_time()}{VERBOSE_STRING} ADDING NEW DATA TO {new_image_file} [MAIN()]")
    im.putdata(pixels)
    if args.verbose and not args.quiet:
        print(f"{current_time()}{VERBOSE_STRING} SAVING {new_image_file} [MAIN()]")
    im.save(new_image_file)

    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} SAVED TO"\
                f"[underline]{new_image_file}[/underline] [MAIN()]")
        else:
            print(f"{DONE} Saved to [underline]{new_image_file}[/underline]")
    if not args.quiet:
        print(f"{current_time()} Finished")
try:
    if __name__ == "__main__":
        main()
except KeyboardInterrupt:
    print("Exiting...")
