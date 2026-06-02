from PIL import Image, ImageTk, ImageEnhance, ImageFilter # img processing
import numpy as np # better data handling and L2 norm calculation
import cv2
import tkinter as tk
from skimage import color as skcolor
import time


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


def recolour_img(inputted_img: Image.Image, neutral_penalty: float = 12.0) -> Image.Image:
    rgb_palette = np.array(list(wplace_palette.values()), dtype=np.float32) / 255.0

    # Convert to lab for channel enhancement on dull colours
    lab_palette = skcolor.rgb2lab(rgb_palette[None, :, :])[0]
    palette_chroma = np.sqrt(lab_palette[:, 1] ** 2 + lab_palette[:, 2] ** 2)

    # Penalise colours that are low chroma but not pure black/white
    L = lab_palette[:, 0]
    low_chroma_threshold = (palette_chroma < 15.0) & (L > 8.0) & (L < 95.0)
    penalty = np.where(low_chroma_threshold, neutral_penalty, 0.0)

    # Convert image pixels to LAB
    pixels = np.array(inputted_img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = pixels.shape[:2]
    lab_pixels = skcolor.rgb2lab(pixels)

    # Compute shortest distance when accounting for the penalty imposed for low chroma threshold
    flat = lab_pixels.reshape(-1, 3)
    dists = np.linalg.norm(flat[:, None, :] - lab_palette[None, :, :], axis=2)
    dists = dists + penalty[None, :]
    closest_idx = np.argmin(dists, axis=1)

    # Map the closest colours
    rgb_palette_uint8 = np.array(list(wplace_palette.values()), dtype=np.uint8)
    processed_array = rgb_palette_uint8[closest_idx].reshape(h, w, 3)

    return Image.fromarray(processed_array, "RGB")


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


def sobel_mapping(inputted_img: np.ndarray) -> np.ndarray:
    # Sobel operator requires grayscale

    # Blur to help reduce noise
    img_blurred = cv2.GaussianBlur(inputted_img, (3, 3), 0)

    # Compute Sobel operator for x and y
    sobel_x = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Convert back to integers. Use L1 norm
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # Combine the gradients
    sobel_combined = cv2.addWeighted(sobel_x_abs,0.5, sobel_y_abs,0.5,0)

    return sobel_combined


def edge_mapping(inputted_img: Image.Image, sobel_matrix: np.ndarray, edge_strength: float = 0.6):
    pixels = np.array(inputted_img.convert("RGB")).astype(float)

    # Normalise sobel to [0, 1], broadcasted over RGB channels
    sobel_norm = (sobel_matrix / 255.0)[:, :, None]

    # Scale the strength of edge based on sobel to determine how black to make the edge
    soft_mask = np.clip((sobel_norm - (threshold / 255.0)) * 4.0, 0.0, 1.0) * edge_strength
    edge_colour = np.array([0, 0, 0], dtype=float)
    blended = pixels * (1 - soft_mask) + edge_colour * soft_mask

    # Convert back to RGB
    processed_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")
    return processed_img


def compress_img(
        inputted_img: Image.Image,
        target_pixels: int,
        sobel_matrix: np.ndarray,
        flat_preserve: float = 0.8
    ) -> Image.Image:

    w, h = inputted_img.size

    # In case image is way too small
    block_size = max(1, int(np.sqrt((w * h) / target_pixels)))
    pixels = np.array(inputted_img.convert("RGB")).astype(float)
    new_h = h // block_size
    new_w = w // block_size

    # Duplicate pixel and sobel matrices but with their smaller size
    pixels_cropped = pixels[:new_h * block_size, :new_w * block_size]
    sobel_cropped  = sobel_matrix[:new_h * block_size, :new_w * block_size].astype(float)

    # Reshape and take the average for pixel and sobel blocks
    blocks = pixels_cropped.reshape(new_h, block_size, new_w, block_size, 3)
    block_mean = blocks.mean(axis=(1, 3))
    sobel_blocks = sobel_cropped.reshape(new_h, block_size, new_w, block_size)
    edge_strength = sobel_blocks.mean(axis=(1, 3)) / 255.0

    # Extract centre pixel of each block
    mid = block_size // 2
    centre = pixels_cropped[mid::block_size, mid::block_size]
    centre = centre[:new_h, :new_w]

    # Broadcast the edge strength over the RGB channels
    e = edge_strength[:, :, None]
    output = e * block_mean + (1 - e) * (centre * flat_preserve + block_mean * (1 - flat_preserve))

    compressed = Image.fromarray(output.astype(np.uint8), "RGB")
    return compressed.resize((new_w * block_size, new_h * block_size), Image.NEAREST)


def dither_bayer(
        inputted_img: Image.Image,
        strength: float = 30.0,
        matrix: np.ndarray = None,
        chroma_boost: float = 2.5,
        chroma_threshold: float = 20.0
    ) -> Image.Image:

    pixels = np.array(inputted_img.convert("RGB")).astype(np.float32) / 255.0
    h, w = pixels.shape[:2]
    m = matrix.shape[0]

    tiled = np.tile(matrix, (h // m + 1, w // m + 1))[:h, :w]

    # Convert to lab to boost channels
    lab = skcolor.rgb2lab(pixels)

    # Use chroma magnitude to determine where boost is needed
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    t = np.clip(1.0 - (chroma / chroma_threshold), 0.0, 1.0)
    boost = 1.0 + (chroma_boost - 1.0) * t

    # Apply dither. L takes base, while a/b take boosted to emphasize not throwing out lowly saturated colours
    noise = tiled * strength
    lab[:, :, 0] += noise
    lab[:, :, 1] += noise * boost
    lab[:, :, 2] += noise * boost

    # Convert back to RGB
    perturbed_rgb = skcolor.lab2rgb(np.clip(lab,[0, -128, -128],[100, 127, 127]))
    perturbed_uint8 = np.clip(perturbed_rgb * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(perturbed_uint8, "RGB")


start_time = time.perf_counter()

# Open image file
img: Image.Image = Image.open("makima1.png")

THRESHOLD = 40

# Bayer matrix for dithering
BAYER_4x4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=float) / 16.0 - 0.5


EDGE_STRENGTH = 0.60
DITHER_STRENGTH = 25.0

# Preprocess -> Edge mapping -> Compress -> Dither -> Recolour
preprocessed_img = preprocess_img(img, saturation=1.2, shadow_contrast=1.2)
img_cv2_grayscale: np.ndarray = cv2.cvtColor(np.array(preprocessed_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
sobel_matrix = sobel_mapping(img_cv2_grayscale)
edge_img = edge_mapping(preprocessed_img, sobel_matrix)
compressed_img = compress_img(edge_img, target_pixels=58_000, sobel_matrix=sobel_matrix)
dithered_img = dither_bayer(compressed_img, strength=DITHER_STRENGTH, matrix=BAYER_4x4)
final_img = recolour_img(dithered_img)


final_img.save("bruh.png")

end_time = time.perf_counter()

print(f"Total time taken: {end_time - start_time:.4f} seconds")

