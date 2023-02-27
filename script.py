"""module for manipulating image data in order for getting interesting results"""
#!/usr/bin/env python3

import random
import argparse
import os.path
from datetime import datetime
from typing import List
from pathlib import Path
from tqdm import tqdm # pylint: disable=import-error
from rich import print # pylint: disable=redefined-builtin, import-error
from PIL import Image, ImageChops


start_time = datetime.now()

VERBOSE_STRING = "[yellow][VERBOSE][/yellow]"

CORRECT = "[green][*][/green]"
DONE    = "[green][DONE][/green]"
NOTICE  = "[yellow][!][/yellow]"

IMAGE_SIMULATION_SIZE = list(range(2795520))
#TODO: maybe edit metadata to show what manipulations has been done to the image
#TODO: allow multiple files to be given from command line and loop over each file
#TODO: add flags for each image filter


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
                    help="[!DEPRECATED!] force output to the default location"
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
parser.add_argument(
                    "-s","--simulate",
                    dest="simulate",
                    action="store_true",
                    help="[!NOT YET IMPLEMENTED!] Simulates working on an image"
                    )
args = parser.parse_args()

class TqdmWrapper:
    """Wrapper class for the tqdm progress bar"""
    def __init__(self, desc="DESC MISSING", total=None, disable=False, quiet=False):
        self.desc = desc
        self.total = total
        self.disable = disable or quiet
        self.bar_styles = {
            "builder": " ▖▘▝▗▚▞█",
            "fade": "░▒█",
            "arrow": " >=",
        }
    def __enter__(self):
        self.pbar = tqdm(
            total=self.total,
            desc=self.desc,
            ascii=self.bar_styles["fade"],
            bar_format="{desc}: {percentage:3.0f}% |{bar}| ETA: [{remaining}]"
                    if not args.verbose else
                        "{l_bar}{bar}| [{n_fmt}/{total_fmt} Total Iterations "\
                        "| {rate_fmt} | ETA: {eta:%y-%m-%d %H:%M}{postfix} "\
                        "| ETR: {remaining}]",
            leave=args.verbose,#False if not args.verbose else True,
            disable=self.disable
        )
        return self.pbar

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()

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

    NOTE:
    -----
        * If both bools are False or True, both time and date are returned
    """
    if just_time:
        return datetime.now().strftime("[%H:%M:%S]")
    if just_date:
        return datetime.now().strftime("[%d/%m/%Y]")
    if just_date and just_time :
        return datetime.now().strftime("[%d/%m/%Y|%H:%M:%S]")
    else:
        return datetime.now().strftime("[%d/%m/%Y|%H:%M:%S]")

def noise(image:Image, intensity:int = 10) -> List:
    """
    Adds pseudo noise to an image
    PARAMS:
    -------
        * image `Image`: Input image
        * intensity `int`: Noise intensity
    RETURNS:
    --------
        * _new_pixels `list`: a list of pixels with added noise
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} ADDING NOISE TO {image.filename} "\
                    f"[NOISE(image={image.filename}, {intensity=})]")
        else:
            print(f"{CORRECT} Adding noise")
    img_pixels = list(image.getdata())
    _new_pixels = []
    #for pixel in tqdm(img_pixels,total=len(img_pixels),desc="Adding noise",bar_format="{desc}: {percentage:3.0f}% |{bar}|",leave=False,disable=args.no_progress or args.quiet):
    with TqdmWrapper(
                    desc="Adding Noise",
                    total=len(img_pixels),
                    disable=args.no_progress or args.quiet
                    ) as pbar:
        for pixel in img_pixels:
            # RNG for adding or subtracting pixel brightness
            pixel_brightness_rng = random.randint(1,2)
            # RNG to determine if a pixel should be manipulated
            manipulate_pixel_rng = random.randint(1,100)
            if manipulate_pixel_rng < intensity:
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
            pbar.update(1)
    return _new_pixels


def shift(image:Image, intensity:int=10) -> List:
    """
    Shifts random pixels of an image

    PARAMS:
    -------
        * image `Image`: Input image
        * intensity `int`: Shifting intensity

    RETURNS:
    --------
        * _new_pixels `list`: A list of shifted pixels
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "\
                    f"ADDING PIXEL SHIFTING TO {image.filename} [SHIFT(image={image.filename}, {intensity=})]")
        else:
            print(f"{CORRECT} Shifting pixels")
    width, height = image.size
    img_pixels = list(image.getdata()) # RGB data of each pixel
    _new_pixels = []
    with TqdmWrapper(
                    desc="Shifting",
                    total=height,
                    disable=args.no_progress or args.quiet
                    ) as pbar:
        for _y in range(height):
            # Generate a random integer between 1 and 100
            manipulate_pixel_chance_rng = random.randint(1,100)
            # Generate a random integer between 0 and the width of the image
            shift_distance_rng = random.randint(0,width)
            for _x in range(width):
                # checks if the pixels is to be shifted
                if manipulate_pixel_chance_rng < intensity:
                    # Calculate the new x and y positions for the shifted pixel
                    new_x = (_x + shift_distance_rng) % width
                    new_y = (_y + shift_distance_rng) % height
                    # Append the shifted pixel to the new_pixels list
                    _new_pixels.append(img_pixels[new_y * width + new_x])
                else:
                    # Append the original pixel to the new_pixels list
                    _new_pixels.append(img_pixels[_y * width + _x])
            pbar.update(1)
    return _new_pixels



def duplicate(image: Image, grid_size:int=4, chance:int=5) -> List:
    """
    CURRENTLY DOESNT WORK ALONE, OTHER FUNCTIONS MUST BE PASSED TO COMBINE_PIXELS ALONGSIDE THIS ONE!
    Duplicates random NxN grid of pixels of an image and places them at a random location
    
    PARAMS:
    -------
        * image `Image`: Input image
        * grid_size `int`: Size of the grid to be shifted. Defaults to 4
        * chance `int`: Chance of a pixel to be shifted. Defaults to 5

    RETURNS:
    --------
        * _new_pixels `list`: A list of duplicated pixels

    NOTE:
    -----
        * the bigger the grid_size the longer the function takes

    
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "
                    f"DUPLICATING PIXELS TO {image.filename} [DUPLICATE(image={image.filename}, {grid_size=}, {chance=})]")
        else:
            print(f"{CORRECT} Duplicating pixels")

    width, height = image.size
    img_pixels = list(image.getdata()) # RGB data of each pixel
    
    _new_pixels = []
    with TqdmWrapper(
                    desc="Duplicating",
                    total=height,
                    disable=args.no_progress or args.quiet
                    ) as pbar:
        for _y in range(height):
            for _x in range(width):
                manipulate_pixel_chance_rng = random.randint(1, 100)
                # checks if the pixels is to be duplicated
                if manipulate_pixel_chance_rng < chance:
                    # Generate a random integer between 0 and the width-grid_size of the image
                    x_start_rng = random.randint(0, width  - grid_size)
                    y_start_rng = random.randint(height//2, height - grid_size)
                    for new_y in range(y_start_rng,     y_start_rng + grid_size):
                        for new_x in range(x_start_rng, x_start_rng + grid_size):
                            # Append the duplicated pixel to the new_pixels list
                            _new_pixels.append(img_pixels[new_y * width + new_x])
                else:
                    #pass
                    _new_pixels.append(img_pixels[_y * width + _x])
            pbar.update(1)
    return _new_pixels


def chromatic_aberration(image: Image, shift_size: int = 1) -> List:
    """
    Shifts the color channels of an image to create chromatic aberration.

    PARAMS:
    -------
        * image `Image`: Input image.
        * shift_size `int`: Number of pixels to shift the color channels. Defaults to 10.

    RETURNS:
    --------
        * _new_pixels `list`: List of shifted pixels.
    """

    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "\
                    f"ADDING CHROMATIC ABERRATION TO {image.filename} "\
                    f"[CHROMATIC_ABERRATION(image={image.filename},{shift_size=})]")
        else:
            print(f"{CORRECT} Adding chromatic aberration")

    # Split the image into its color channels.
    red, green, blue = image.split()

    # Shift the color channels in different directions.
    red_shifted   = ImageChops.offset(red, shift_size, shift_size)
    green_shifted = ImageChops.offset(green, 0, -shift_size)
    blue_shifted  = ImageChops.offset(blue, -shift_size, 0)

    # Merge the shifted color channels back into an image.
    shifted_image = Image.merge("RGB", (red_shifted, green_shifted, blue_shifted))

    # Get the pixels of the shifted image.
    _new_pixels = list(shifted_image.getdata())

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
        if len(pixel_lists) == 0:
            print("[red underline][WARNING][/red underline] "\
                    "[red underline]NO PIXEL LISTS HAVE BEEN GIVEN![/red underline]")
            exit()
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} COMBINING {len(pixel_lists)} "\
                                                "LISTS OF PIXELS [COMBINE_PIXELS()]")
        else:
            print(f"{CORRECT} Combining pixels")
    combined_pixels = []
    with TqdmWrapper(
                    desc="Combining Pixels",
                    total=len(pixel_lists[0]),
                    disable=args.no_progress or args.quiet
                    ) as pbar:
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
            pbar.update(1)
    return combined_pixels

def main():
    """Main entery point"""
    if not args.image:
        print("[red underline][WARNING][/red underline] "\
                f"[red underline]NO IMAGE PATH GIVEN![/red underline]")
        exit()
    if not args.quiet:
        print(f"{current_time()} Starting")

    # path to image (image file included)
    image_file = Path(args.image).absolute()
    # path to folder containing image file
    image_path = Path(image_file.parent)
    file_name = Path(f"new_{image_file.name}")
    if not image_file.exists():
        print(f"[red underline][WARNING][/red underline] "\
                f"[red underline]FILE \"{image_file}\" WAS NOT FOUND[/red underline]")
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

    shift_pixels                = shift(im)
    duplicated_pixels           = duplicate(im)
    noise_pixels                = noise(im, intensity=10)
    chromatic_aberration_pixels = chromatic_aberration(im,shift_size=20)

    pixels = combine_pixels(
                            shift_pixels,
                            duplicated_pixels,
                            noise_pixels,
                            chromatic_aberration_pixels
                            )
    if args.verbose and not args.quiet:
        print(f"{current_time()}{VERBOSE_STRING} ADDING NEW DATA TO {new_image_file} [MAIN()]")
    im.putdata(pixels)
    im.save(new_image_file)

    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} SAVED TO "\
                f"[underline]{new_image_file}[/underline] [MAIN()]")
        else:
            print(f"{DONE} Saved to [underline]{new_image_file}[/underline]")
    if not args.quiet:
        total_seconds = (datetime.now() - start_time).total_seconds()
        hours, remaining_seconds = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remaining_seconds, 60)
        print(f"\n{current_time()} [underline]Finished[/underline]"
                if not args.verbose else
                    f"\n{current_time()} Finished after {int(hours)}:{int(minutes)}:{int(seconds)}")

try:
    if __name__ == "__main__":
        main()
except KeyboardInterrupt:
    print("Exiting...")
