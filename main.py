# Project goal: implement K-means clustering to compress an image into a certain
# number of pixels to be plotted on wplace. There are three main components to this

# 1) Colour designation: there are only a certain number of colours available in
#    the wplace palette, so we need to convert colours to match whe colours

# 2) Image compression: while theoretically possible, mapping every single pixel
#    in a high definition image is extremely tedious on wplace, so we will need to
#    be able to compress the photo to match a designated area (say 10k pixels)

# 3) Image segmentation: standard compression will lump pixels together based on
#    simple distances (like euclidean distance) but this is not good if we are
#    trying to clump together pieces that are clearly part of a cluster (i.e;
#    foreground vs. background) so we need to have segmentation incorporated
#    prior to compression

# Let's start with the simplest to implement; colour designation. This is simply
# a matter of finding the closest colour (we can use L2 norm) and reassigning that
# pixel first.

# Colours are designated based on name, so let's store the hexa values in a dict

from PIL import Image, ImageEnhance, ImageFilter # img processing
import numpy as np # better data handling and L2 norm calculation
import cv2
from scipy.ndimage import sobel


# The palette (in RGB) for wplace is the following:
wplace_palette = {
    "Black": (0, 0, 0),
    "Dark Gray": (60, 60, 60),
    "Gray": (120, 120, 120),
    "Medium Gray": (170, 170, 170),
    "Light Gray": (210, 210, 210),
    "White": (255, 255, 255),

    "Deep Red": (96, 0, 24),
    "Dark Red": (165, 14, 30),
    "Red": (237, 28, 36),
    "Light Red": (250, 128, 114),

    "Dark Orange": (228, 92, 26),
    "Orange": (255, 127, 39),

    "Gold": (246, 170, 9),
    "Yellow": (249, 221, 59),
    "Light Yellow": (255, 250, 188),

    "Dark Goldenrod": (156, 132, 49),
    "Goldenrod": (197, 173, 49),
    "Light Goldenrod": (232, 212, 95),

    "Dark Olive": (74, 107, 58),
    "Olive": (90, 148, 74),
    "Light Olive": (132, 197, 115),

    "Dark Green": (14, 185, 104),
    "Green": (19, 230, 123),
    "Light Green": (135, 255, 94),

    "Dark Teal": (12, 129, 110),
    "Teal": (16, 174, 166),
    "Light Teal": (19, 225, 190),

    "Dark Cyan": (15, 121, 159),
    "Cyan": (96, 247, 242),
    "Light Cyan": (187, 250, 242),

    "Dark Blue": (40, 80, 158),
    "Blue": (64, 147, 228),
    "Light Blue": (125, 199, 255),

    "Dark Indigo": (77, 49, 184),
    "Indigo": (107, 80, 246),
    "Light Indigo": (153, 177, 251),

    "Dark Slate Blue": (74, 66, 132),
    "Slate Blue": (122, 113, 196),
    "Light Slate Blue": (181, 174, 241),

    "Dark Purple": (120, 12, 153),
    "Purple": (170, 56, 185),
    "Light Purple": (224, 159, 249),

    "Dark Pink": (203, 0, 122),
    "Pink": (236, 31, 128),
    "Light Pink": (243, 141, 169),

    "Dark Peach": (155, 82, 73),
    "Peach": (209, 128, 120),
    "Light Peach": (250, 182, 164),

    "Dark Brown": (104, 70, 52),
    "Brown": (149, 104, 42),
    "Light Brown": (219, 164, 99),

    "Dark Tan": (123, 99, 82),
    "Tan": (156, 132, 107),
    "Light Tan": (214, 181, 148),

    "Dark Beige": (209, 128, 81),
    "Beige": (248, 178, 119),
    "Light Beige": (255, 197, 165),

    "Dark Stone": (109, 100, 63),
    "Stone": (148, 140, 107),
    "Light Stone": (205, 197, 158),

    "Dark Slate": (51, 57, 65),
    "Slate": (109, 117, 141),
    "Light Slate": (179, 185, 209),
}

# Find the closest colour for each pixel
def closest_colour(current_colour):
    # Starting from max, find the smallest distance from the actual colour of
    # the pixel to each of the possible colours in the palette
    current_min = float("inf")
    approx_colour = None

    for target_colour in wplace_palette.values():
        target_colour_dist = np.linalg.norm(
            np.array(target_colour) - np.array(current_colour)
        )
        if target_colour_dist < current_min:
            current_min = target_colour_dist
            approx_colour = target_colour

    # Return the best approximation
    return approx_colour


def recolour_img(inputted_img: Image.Image) -> Image.Image:

    rgb_palette = np.array(list(wplace_palette.values()))
    pixels = np.array(inputted_img.convert("RGB"))

    # In order to vectorize correctly, store each pixel in its own row
    flat = pixels.reshape(-1, 3).astype(float)

    # Create a fake dimension for broadcasting. For broadcasting each dimension
    # must have either the same size (idx=2), or one of the dimensions has a size
    # of 1 (we can do this by adding None to idx=0 and idx=1).
    dists = np.linalg.norm(flat[:, None, :] - rgb_palette[None, :, :], axis=2)
    closest_idx = np.argmin(dists, axis=1)

    processed_array = rgb_palette[closest_idx].reshape(pixels.shape)
    processed_img = Image.fromarray(processed_array.astype(np.uint8), "RGB")

    return processed_img


# Implement preprocessing for better saturation and deeper shadows
def preprocess_img(
        inputted_img: Image.Image,
        saturation: float,
        shadow_contrast: float
    ) -> Image.Image:

    # Boost saturation
    saturated_img = ImageEnhance.Color(inputted_img).enhance(saturation)

    # Deepen shadows via contrast
    processed_img = ImageEnhance.Contrast(saturated_img).enhance(shadow_contrast)

    return processed_img

# Open image file
img: Image.Image = Image.open("makima1.png")

# Preprocess image
preprocessed_img = preprocess_img(
    img,
    saturation=1,
    shadow_contrast=1
)

# Process image via recolouring scheme
recoloured_img = recolour_img(preprocessed_img)

# Save image to file
recoloured_img.save("output1.png")



