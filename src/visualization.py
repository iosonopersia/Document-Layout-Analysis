import cv2
from matplotlib import pyplot as plt
from numpy import ndarray

from coco import COCOBoundingBox


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
SILVER = (192, 192, 192)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)
NAVY = (0, 0, 128)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
GRAY = (128, 128, 128)
LIME = (0, 255, 0)
DARK_BLUE = (0, 0, 139)
DARK_CYAN = (0, 139, 139)
DARK_GRAY = (169, 169, 169)
DARK_GREEN = (0, 100, 0)
DARK_MAGENTA = (139, 0, 139)
DARK_RED = (139, 0, 0)
DARK_YELLOW = (139, 139, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_CYAN = (224, 255, 255)
LIGHT_GRAY = (211, 211, 211)
LIGHT_GREEN = (144, 238, 144)
LIGHT_MAGENTA = (255, 182, 193)
LIGHT_RED = (255, 160, 122)
LIGHT_YELLOW = (255, 255, 224)


def normalize_colour(colour: tuple[int]) -> tuple[float]:
    """Normalizes the colour values to be between 0 and 1."""
    return tuple([value / 255.0 for value in colour])


def visualize_bbox(img: ndarray, bbox: COCOBoundingBox, class_name: str, border_colour: tuple[int]=RED,
                   thickness: int=2, is_normalized: bool=True) -> ndarray:
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    assert x_min < x_max and y_min < y_max, "Bounding box is not valid."

    bg_colour = WHITE
    text_colour = BLACK
    if is_normalized:
        border_colour = normalize_colour(border_colour)
        bg_colour = normalize_colour(bg_colour)
        text_colour = normalize_colour(text_colour)

    # Draw the bounding box on the image
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=border_colour, thickness=thickness)

    # Display the label at the top of the bounding box
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), bg_colour, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=text_colour,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image: ndarray, bboxes: list[COCOBoundingBox], category_ids: list[int],
              category_id_to_name: dict[int, str], category_id_to_colour: dict[int, tuple[int]],
              is_normalized: bool=True) -> None:
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        colour = category_id_to_colour[category_id]
        img = visualize_bbox(img, bbox, class_name, colour, is_normalized=is_normalized)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
