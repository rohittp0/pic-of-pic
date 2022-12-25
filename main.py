import glob
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from numba import njit, prange


MAX_SIZE = 2 ** 13


def get_image(path, size=None):
    image = cv2.imread(path)

    if image.shape[0] != image.shape[1]:
        h, w = image.shape[:2]
        side = min(h, w)
        image = image[h // 2 - side // 2:h // 2 + side // 2, w // 2 - side // 2:w // 2 + side // 2]

    if size is not None:
        image = cv2.resize(image, size)

    return image


def get_dominant_color(image):
    a2D = image.reshape(-1, image.shape[-1])
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)

    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def get_dominant_color_from_path(path, image_size):
    image = get_image(path, (image_size, image_size))
    return get_dominant_color(image)


def get_dominant_colors(image_path, image_size):
    paths = glob.glob(f"{image_path}/*")

    with Pool() as pool:
        result = pool.map(
            partial(get_dominant_color_from_path, image_size=image_size),
            paths
        )

    return np.array(result), paths


@njit(parallel=True)
def closest_color(colors, color):
    distances = np.sqrt(np.sum(np.power(colors - color, 2), axis=1))
    return np.argmin(distances)


def get_output_image(image_size, paths, index):
    return get_image(paths[index], (image_size, image_size))


def get_color_of_region(image, dom_colors, index):
    i, j = index
    color = get_dominant_color(image[i:i + 1, j:j + 1])
    index = closest_color(dom_colors, np.array(color))

    return index


@njit(parallel=True)
def arrange_images(images, image_size, output_size):
    output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    lim = int(output_size / image_size)

    for i in prange(lim):
        for j in prange(lim):
            r = i * image_size
            c = j * image_size

            output[r: r + image_size, c:c + image_size] = images[i * lim + j]

    return output


def main():
    image_path = "images/varsha"
    target = "images/target.jpg"

    partition_size = 20
    image_size = 64

    dom_colors, paths = get_dominant_colors(image_path, image_size)

    target_image = get_image(target)
    target_size = target_image.shape[0] / partition_size
    target_size = round(min(MAX_SIZE / image_size, target_size))

    target_image = cv2.resize(target_image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    print("Got dominant colors")

    ranges = list(range(0, target_size))
    ranges = [(i, j) for i in ranges for j in ranges]

    with Pool() as pool:
        ranges = pool.map(partial(get_color_of_region, target_image, dom_colors), ranges)

    print("Got color of", len(ranges), "regions")
    del target_image, dom_colors

    with Pool() as pool:
        result = pool.map(
            partial(get_output_image, image_size, paths),
            ranges
        )

    result = np.array(result)
    del paths, ranges

    print("Got output images")

    output = arrange_images(result, image_size, int(target_size * image_size))

    cv2.imwrite("output.png", output)


if __name__ == "__main__":
    main()
