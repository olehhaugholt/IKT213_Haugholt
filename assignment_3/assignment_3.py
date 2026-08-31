import cv2
import numpy as np

def sobel_edge_detection(image):
    """
    Sobel edge detection on an input image
    """

    # Load image and convert to grayscale with blur
    image_color = cv2.imread(image)
    image_grayscale = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_grayscale, (3, 3), 0)

    # Sobel edge detection
    sobel_x = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_xy = cv2.magnitude(sobel_x, sobel_y)

    # normalize before saving
    sobel_xy = cv2.normalize(sobel_xy, None, 0, 255, cv2.NORM_MINMAX)
    sobel_xy = sobel_xy.astype(np.uint8)

    # save the output image
    cv2.imwrite("./assignment_3/solutions/output_sobel.jpg", sobel_xy)


def canny_edge_detection(image):
    """
    Canny edge detection on an input image
    """

    # Load image and convert to grayscale with blur
    image_color = cv2.imread(image)
    image_grayscale = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_grayscale, (3, 3), 0)

    # Canny edge detection
    canny_edges = cv2.Canny(image_blurred, 50, 50)

    # normalize before saving
    canny_edges = cv2.normalize(canny_edges, None, 0, 255, cv2.NORM_MINMAX)
    canny_edges = canny_edges.astype(np.uint8)

    # save the output image
    cv2.imwrite("./assignment_3/solutions/output_canny.jpg", canny_edges)


def template_match(image, template):
    """
    Use template matching to find a template in an input image
    """

    # Load image and template
    image_color = cv2.imread(image)
    template_color = cv2.imread(template)

    # Convert to grayscale
    image_grayscale = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    template_grayscale = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

    w, h = template_grayscale.shape[::-1]

    # Apply template matching
    res = cv2.matchTemplate(image_grayscale, template_grayscale, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw in rectangle around the matched area
    threshold = 0.9
    cv2.rectangle(image_color, top_left, bottom_right, (0, 0, 255), 2)

    # Normalize for visualization
    res_visual = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    res_visual = res_visual.astype(np.uint8)

    # save the output image
    cv2.imwrite("./assignment_3/solutions/output_template_matching.jpg", image_color)
    cv2.imwrite("./assignment_3/solutions/output_template_matching_result.jpg", res_visual)


def resize(image, scale_factor: int, up_or_down: str):
    """
    Resize an image by a scale factor
    """

    # Load image
    image_color = cv2.imread(image)

    # Get the dimensions of the image
    rows, cols, _channels = map(int, image_color.shape)

    # Resize the image
    if up_or_down == "up":
        resized_image = cv2.pyrUp(image_color, dstsize=(scale_factor * cols, scale_factor * rows))
    elif up_or_down == "down":
        resized_image = cv2.pyrDown(image_color, dstsize=(cols // scale_factor, rows // scale_factor))
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    # save the output image
    cv2.imwrite(f"./assignment_3/solutions/output_resize.jpg", resized_image)


# lambo image
img_lambo = "/Users/olehh/UiA/IKT213_Haugholt/assignment_3/lambo.png"

# shapes image
img_shapes = "/Users/olehh/UiA/IKT213_Haugholt/assignment_3/shapes-1.png"
img_shapes_template = "/Users/olehh/UiA/IKT213_Haugholt/assignment_3/shapes_template.jpg"

sobel_edge_detection(img_lambo)
canny_edge_detection(img_lambo)
resize(img_lambo, 2, "up")
template_match(img_shapes, img_shapes_template)