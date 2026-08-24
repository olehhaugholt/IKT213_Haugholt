import cv2
import numpy as np
import matplotlib.pyplot as plt


def padding(image, border_width):
    """
    1. Padding: make border around the image
    """
    padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite('assignment_2/solutions/iris-padded.jpg', padded)

    return padded 

def crop(image, x_0, x_1,  y_0, y_1):
    """
    2. Cropping: crop the image to a specific region
    """
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite('assignment_2/solutions/iris-cropped.jpg', cropped)

    return cropped

def resize(image, width, height):
    """
    3. Resize: resize the image to a specific size
    """
    resized = cv2.resize(image, (width, height))
    cv2.imwrite('assignment_2/solutions/iris-resized.jpg', resized)

    return resized

def copy(image, empty_picture_array):
    """
    4. Copy: create a copy of the image
    """
    np.copyto(empty_picture_array, image)
    cv2.imwrite('assignment_2/solutions/iris-copied.jpg', empty_picture_array)

    return empty_picture_array

def grayscale(image):
    """
    5. Color to Greyscale
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('assignment_2/solutions/iris-grayscale.jpg', gray)

    return gray

def hsv(image):
    """
    6. Color to HSV
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('assignment_2/solutions/iris-hsv.jpg', hsv_image)

    return hsv_image

def hue_shifted(image, empty_picture_array, shift_value):
    """
    7. Hue Shift: shift the hue of the image by a specific value
    """
    np.copyto(empty_picture_array, image)
    empty_picture_array += shift_value
    cv2.imwrite('assignment_2/solutions/iris-hue-shifted.jpg', empty_picture_array)

    return empty_picture_array

def smoothing(image, kernel_size):
    """
    8. Smoothing: apply a blur to the image
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    cv2.imwrite('assignment_2/solutions/iris-smoothed.jpg', blurred)

    return blurred

def rotation(image, angle):
    """
    9. Rotation: rotate the image by a specific angle
    """
    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)

    cv2.imwrite('assignment_2/solutions/iris-rotated-180.jpg', rotated)
    
    return rotated

# load the original image
image = cv2.imread('assignment_2/iris-1.jpg')
padded_image = padding(image, 100)
cropped_image = crop(image, 200, image.shape[1] - 130, 200, image.shape[0] - 130)
resized_image = resize(image, 200, 200)
copied_image = copy(image, np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8))
gray_image = grayscale(image)
hvs_image = hsv(image)
hue_shifted_image = hue_shifted(image, np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8), 50)
smoothed_image = smoothing(image, 15)
rotated_image = rotation(image, 180)

