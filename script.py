"""module for manipulating image data in order for getting interesting results"""
#!/usr/bin/env python3

import random
import argparse
import os
import os.path
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
import math
import shutil
import urllib.parse
from tqdm import tqdm
from rich import print # pylint: disable=redefined-builtin
from PIL import Image, ImageChops
import requests

start_time = datetime.now()

VERBOSE_STRING = "[yellow][VERBOSE][/yellow]"

CORRECT = "[green][*][/green]"
DONE    = "[green][DONE][/green]"
NOTICE  = "[yellow][!][/yellow]"

DOWNLOADED_FOLDER_PATH = Path(os.path.join(Path.cwd(), "downloaded"))


#TODO: maybe edit metadata to show what manipulations has been done to the image
#TODO: allow multiple files to be given from command line and loop over each file
#TODO: add flags for each image filter

# NOTE: Using Path.cwd() might lead to problems if the script is ran
#       from anywhere else other than here


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
                    "-u", "--url",
                    dest="image_url",
                    action="store",
                    help="url of image"
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
        self.pbar = tqdm( # pylint: disable=attribute-defined-outside-init
            total=self.total,
            desc=self.desc,
            ascii=self.bar_styles["fade"],
            bar_format="{desc}: {percentage:3.0f}% |{bar}| ETA: [{remaining}]"
                    if not args.verbose else
                        "{l_bar}{bar}| [{n_fmt}/{total_fmt} Total Iterations "\
                        "| {rate_fmt} | ETA: {eta:%y-%m-%d %H:%M}{postfix} "\
                        "| ETR: {remaining}]",
            leave=args.verbose,
            disable=self.disable
        )
        return self.pbar

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()

def rng(min_num:int=0, max_num:int=100):
    """
    returns a random int between min and max

    PARAMS:
    -------
        * min_num `int`: min number
        * max_num `int`: max number

    RETURNS:
    -------
        * returns a random int between min_num & max_num

    """
    return random.randint(min_num,max_num)

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

def get_files_in_downloaded() -> List[str]:
    """
    Returns the amount of files in the `downloaded` folder
    """
    return os.listdir(DOWNLOADED_FOLDER_PATH)

def clean_downloaded_folder() -> None:
    """
    Cleans the `downloaded` folder
    """
    nl_char = "\n"
    tab_char = "\t"
    print("Removed: "\
            f"\n\tdownloaded/{f'{nl_char}{tab_char}downloaded/'.join(get_files_in_downloaded())}\n")
    for file in get_files_in_downloaded():
        if os.path.exists(file_path:=os.path.join(DOWNLOADED_FOLDER_PATH, file)):
            os.remove(file_path)

def split_url(url:str) -> Tuple:
    """
    Splits a URL

    PARAMS:
    ------
        * url `str`: url string

    RETURNS:
    -------
        * A list of all the URL parts
    """

    parsed_url = urllib.parse.urlsplit(url)
    return parsed_url

def prettify_url(url:str) -> str:
    """
    Prepares url to be downloaded

    PARAMS:
    -------
        * url `str`: Image URL

    RETURNS:
    --------
        * download-ready URL
    """

    return urllib.parse.urlunsplit(url._replace(query="", fragment=""))

def download_image(url:str) -> Path:
    """
    Downloads an image from a URL

    PARAMS:
    ------
        * url `str`: image URL

    RETURNS:
    -------
        * return path to downloaded image
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} DOWNLOADING {url} "\
                    f"[DOWNLOAD_IMAGE(url={url}]")
        else:
            print(f"{CORRECT} Downloading image")

    url = prettify_url(split_url(url))
    online_image_filename = url.split("/")[-1]

    try:
        image_resp = requests.get(url, stream=True, timeout=3)
    except requests.exceptions.ReadTimeout:
        print("[red underline][ERROR][/red underline] "\
            "[red underline]CONNECTION TIMEOUT[/red underline]")
        exit()
    except requests.exceptions.MissingSchema as _e:
        # retrives the corrected link and removes the question mark at the end
        fixed_url = str(_e).rsplit('meant', maxsplit=1)[-1][:-1].strip()
        print("[yellow underline][NOTICE][/yellow underline] Error in URL. Trying again")
        main(fixed_url)

    if image_resp.status_code == 200:
        image_resp.raw.decode_content = True

        if not os.path.exists("downloaded"):
            print("[yellow underline][NOTICE][/yellow underline] \"downloaded\" "\
                "folder not found!. Creating new")
            os.makedirs("downloaded")

        file_size = int(image_resp.headers.get("Content-Length",0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        with tqdm.wrapattr(
            stream=image_resp.raw,
            method="read",
            total=file_size,
            desc=desc,
            leave=args.verbose,
            disable=args.quiet) as r_raw:
            with open(
                    downloaded_image_path:=os.path.join("downloaded",online_image_filename),
                    "wb") as file:
                # remove tqdm.wrapatter & file_size to remove progress bar
                #shutil.copyfileobj(image_resp.raw, file)
                shutil.copyfileobj(r_raw, file)
            return downloaded_image_path
    else:
        print("[red underline][ERROR][/red underline] THE URL COULD NOT BE REACHED")
        return None

def get_image_data(image:Image) -> None:
    """returns basic data about the image"""

    width,height = image.size
    image_name = str(image.filename).rsplit("/", maxsplit=1)[-1]
    image_entropy = image.entropy()
    image_format = image.format
    try:
        animated = image.is_animated
    except AttributeError:
        animated = False
    print(f"{image_format = }")
    print(f"{image_name = }")
    print(f"{width  = }\n{height = }")
    print(f"{image_entropy = }")
    print(f"{animated = }")

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
    with TqdmWrapper(
                    desc="Adding Noise",
                    total=len(img_pixels),
                    disable=args.no_progress or args.quiet
                    ) as pbar:
        for pixel in img_pixels:
            # RNG for adding or subtracting pixel brightness
            pixel_brightness_rng = rng(1,2)
            # RNG to determine if a pixel should be manipulated
            manipulate_pixel_rng = rng(1,100)
            if manipulate_pixel_rng < intensity:
                # Subtract or add random values from the color channels to create a "broken" pixel
                broken_pixel:tuple = (
                                    pixel[0] - rng(0,100)
                                        if pixel_brightness_rng == 1 else
                                            pixel[0] + rng(0,100),
                                    pixel[1] - rng(0,100)
                                        if pixel_brightness_rng == 1 else
                                            pixel[1] + rng(0,100),
                                    pixel[2] - rng(0,100)
                                        if pixel_brightness_rng == 1 else
                                            pixel[2] + rng(0,100)
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
                    f"ADDING PIXEL SHIFTING TO {image.filename} "\
                    f"[SHIFT(image={image.filename}, {intensity=})]")
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
            manipulate_pixel_chance_rng = rng(1,100)
            # Generate a random integer between 0 and the width of the image
            shift_distance_rng = rng(0,width)
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
        * The bigger the grid_size the longer the function takes
        * Cannot be passed to combine_pixels() alone
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "\
                    f"DUPLICATING PIXELS TO {image.filename} "\
                    f"[DUPLICATE(image={image.filename}, {grid_size=}, {chance=})]")
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
                manipulate_pixel_chance_rng = rng(1, 100)
                # checks if the pixels is to be duplicated
                if manipulate_pixel_chance_rng < chance:
                    # Generate a random integer between 0 and the width-grid_size of the image
                    x_start_rng = rng(0, width  - grid_size)
                    y_start_rng = rng(height//2, height - grid_size)
                    for new_y in range(y_start_rng,     y_start_rng + grid_size):
                        for new_x in range(x_start_rng, x_start_rng + grid_size):
                            # Append the duplicated pixel to the new_pixels list
                            _new_pixels.append(img_pixels[new_y * width + new_x])
                else:
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
    NOTE:
    _____
        * If shift_size is 0 it will be a random distance in any direction.
    """

    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "\
                    f"ADDING CHROMATIC ABERRATION TO {image.filename} "\
                    f"[CHROMATIC_ABERRATION(image={image.filename}, {shift_size=})]")
        else:
            print(f"{CORRECT} Adding chromatic aberration")

    # Split the image into its color channels.
    red, green, blue = image.split()
    if shift_size:
        # Shift the color channels in different directions.
        red_shifted   = ImageChops.offset(red, shift_size, shift_size)
        green_shifted = ImageChops.offset(green, 0, -shift_size)
        blue_shifted  = ImageChops.offset(blue, -shift_size, 0)
    else:
        shift_size = rng(1,100)
        red_shifted   = ImageChops.offset(red, rng(1,255), rng(1,255))
        green_shifted = ImageChops.offset(green, rng(1,100), -rng(1,255))
        blue_shifted  = ImageChops.offset(blue, -rng(1,255), rng(1,100))

    # Merge the shifted color channels back into an image.
    shifted_image = Image.merge("RGB", (red_shifted, green_shifted, blue_shifted))

    # Get the pixels of the shifted image.
    _new_pixels = list(shifted_image.getdata())

    return _new_pixels

def vignette(image: Image, intensity: int = 1):
    """
    Adds a vignette to a picture

    PARAMS:
    -------
        * image `Image`: Input image.
        * intensity `int`: Number of pixels to shift the color channels. Defaults to 10.

    RETURNS:
    --------
        * _new_pixels `list`: List of shifted pixels.
    """
    if not args.quiet:
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} "\
                    f"ADDING VIGNETTE TO {image.filename} "\
                    f"[VIGNETTE(image={image.filename}, {intensity=})]")
        else:
            print(f"{CORRECT} Adding Vignette")

    width, height = image.size
    pixels = list(image.getdata())
    new_intensity = min(intensity, 100)
    if intensity == 0:
        return pixels
    _new_pixels = []
    with TqdmWrapper(
                    desc="Adding vignette",
                    total=height,
                    disable=args.no_progress or args.quiet
                    ) as pbar:
        for _y in range(height):
            for _x in range(width):
                # current pixel value
                pixel = pixels[_y * width + _x]
                # distance to center from current pixel
                distance = math.sqrt((_x - width / 2) ** 2 + (_y - height / 2) ** 2)
                # intensity factor based on the distance from the center
                intensity_factor = 1-(distance / (math.sqrt((width / 2) ** 2 + (height / 2) ** 2)))
                # adds intensity factor to the pixel values
                new_pixel = (
                    int(pixel[0] * new_intensity / intensity),
                    int(pixel[1] * new_intensity / intensity),
                    int(pixel[2] * new_intensity / intensity),
                            )
                # adds vignette by multiplying the pixel values by the intensity factor
                new_pixel = (
                    int(new_pixel[0] * intensity_factor),
                    int(new_pixel[1] * intensity_factor),
                    int(new_pixel[2] * intensity_factor),
                            )
                _new_pixels.append(new_pixel)
            pbar.update(1)
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

def main(url=None):
    """Main entery point"""
    # if both output flags are enabled
    if args.quiet and args.verbose:
        #args.verbose = False
        choice = input("Both the \"quiet\" and the \"verbose\" flag are set to True."\
                        "disable \"quiet\" flag? [Y/n]").strip().lower() or ""
        if choice == "" or choice == "y":
            args.verbose = True
            args.quiet = False
    if not args.quiet and not url:
        print(f"{current_time()} Starting")

    # Checks files in downloaded and asks user if they want the deleted
    if args.verbose and url is None:
        print(f"{current_time()}{VERBOSE_STRING} CHECKING FOLDERS")
    if not os.path.exists("output"):
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} CREATING OUTPUT FOLDER")
        os.makedirs("output")
    if not os.path.exists("downloaded"):
        if args.verbose:
            print(f"{current_time()}{VERBOSE_STRING} CREATING DOWNLOADED FOLDER")
        os.makedirs("downloaded")

    # folder clean up
    if len(get_files_in_downloaded()) >= 10:
        print("[yellow underline][NOTICE][/yellow underline] "\
                f"[yellow underline]There are {len(get_files_in_downloaded())} "\
                "files in the \"downloaded\" folder.[/yellow underline]")
        print("[yellow underline]Do you want to remove them? [Y/n]: [/yellow underline]", end="")

        choice = input().strip().lower()
        if choice == "y" or choice == "":
            clean_downloaded_folder()

    if url:
        args.image_url = url
    if args.image_url and args.image:
        print("[red underline][WARNING][/red underline] "\
                "[red underline]CANT USE ONLINE IMAGE URL "\
                "AND LOCAL IMAGE FILE AT THE SAME TIME![/red underline]")
        exit()
    if not args.image and not args.image_url:
        print("[red underline][WARNING][/red underline] "\
                "[red underline]NO IMAGE PATH OR URL GIVEN![/red underline]")
        exit()

    if args.image_url:
        # downloads image from url
        image_file = download_image(args.image_url)
        if image_file is None:
            print("[red underline][ERROR][/red underline] "\
                    "[red underline]IMAGE IS NONE[/red underline]")
            exit()
        image_file = Path(image_file)
    elif args.image:
        # path to image (image file included)
        image_file = Path(args.image).absolute()
    else:
        print("[red underline][ERROR][/red underline]")
    # path to folder containing image file
    image_path = Path(image_file.parent)
    file_name = Path(f"new_{image_file.name}")
    if not image_file.exists():
        print("[red underline][WARNING][/red underline] "\
                f"[red underline]FILE \"{image_file}\" WAS NOT FOUND[/red underline]")
        exit()
    if args.output_name:
        file_name = Path(args.output_name)
        print(file_name)
        if not file_name.suffix:
            print("[red underline][WARNING] NO FILE EXTENSION GIVEN! "\
                f"using \"{image_file.suffix}\"[/red underline]")
            file_name = f"{file_name}{image_file.suffix}"

    # path to new image (image file included)
    default_path = Path(os.path.join(Path.cwd(), "output", file_name))
    # path to output folder
    # creates a new default output folder if it isnt already there
    #default_output_path = Path(os.path.join(Path.cwd(), "output"))
    #if not default_output_path.exists():
    #    print(f"{NOTICE} Creating default output folder!")
    #    os.makedirs(default_output_path)
    # checks if working path is the same as image_path
    if image_path == Path.cwd() or args.default_path:
        new_image_file = default_path
        if args.verbose and not args.quiet:
            print(f"{current_time()}{VERBOSE_STRING} USING DEFAULT OUTPUT LOCATION: "\
            f"{new_image_file} [MAIN({url = })]")
    else:
        # deciding if output should be a working dir or image dir
        if not args.output_name:
            # fix for when downloading image
            # "_new" gets added to the start of the file name for some reason
            file_name = str(file_name).replace("new_","")
            new_image_file = Path(os.path.join(Path.cwd(), "output",f"new_{file_name}"))
        elif args.output_name:
            new_image_file = Path(os.path.join(Path.cwd(), "output",f"{args.output_name}"))

    img = Image.open(image_file)
    if args.verbose:
        get_image_data(img)
    try:
        if img.is_animated:
            print("[red underline][WARNING][/red underline] /"
                    "[red underline]IMAGE FILE IS ANIMATED![/red underline]")
    except AttributeError:
        # Only gif files have Image.is_animated
        pass

    shift_pixels                = shift(img)
    duplicated_pixels           = duplicate(img)
    noise_pixels                = noise(img, intensity=10)
    chromatic_aberration_pixels = chromatic_aberration(img, shift_size=10)
    vignette_pixels             = vignette(img, intensity=100)

    pixels = combine_pixels(
                            shift_pixels,
                            duplicated_pixels,
                            noise_pixels,
                            chromatic_aberration_pixels,
                            vignette_pixels,
                            )
    if args.verbose and not args.quiet:
        print(f"{current_time()}{VERBOSE_STRING} "\
                f"ADDING NEW DATA TO {new_image_file} [MAIN({url=})]")
    img.putdata(pixels)
    if args.verbose and not args.quiet:
        print(f"{current_time()}{VERBOSE_STRING} SAVING {new_image_file} [MAIN({url=})]")
    img.save(new_image_file)

    if args.verbose:
        print(f"{current_time()}{VERBOSE_STRING} SAVED TO "\
                f"[underline]{new_image_file}[/underline] [MAIN({url=})]")
        total_seconds = (datetime.now() - start_time).total_seconds()
        hours, remaining_seconds = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remaining_seconds, 60)
        print(f"\n{current_time()}{VERBOSE_STRING} "\
                f"Finished after {int(hours)}:{int(minutes)}:{int(seconds)}")
    elif not args.verbose and not args.quiet:
        print(f"{DONE} Saved to [underline]{new_image_file}[/underline]")
        print(f"\n{current_time()} Finished")
    # Hacky fix for when the original url doesnt contain "http" and requests tries to fix it.
    # requests returns a guess at the fixed url and I pass it to main().
    # after main() is done with the fixed URL, an UnboundLocalError exception is raised
    # for image_resp from download_image().
    # so exit() is a quick and dirty fix for that exception.
    exit()


try:
    if __name__ == "__main__":
        main()
except KeyboardInterrupt:
    print("Exiting...")
