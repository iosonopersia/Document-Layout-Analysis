from dataclasses import dataclass


@dataclass(frozen=True)
class COCOBoundingBox:
    x: float # Top left corner
    y: float # Top left corner
    w: float # Width
    h: float # Height
    max_width: int = 1 # These are used to scale the bounding box to the image size
    max_height: int = 1 # These are used to scale the bounding box to the image size

    def __post_init__(self):
        """
        Checks that the bounding box is valid

        Raises:
            ValueError: If the bounding box is invalid
        """
        # ==========UNFIXABLE PROBLEMS==========
        if self.max_width < 1 or self.max_height < 1:
            raise ValueError("Max width and height must be greater than 1")
        if self.x > self.max_width or self.y > self.max_height:
            raise ValueError("x and y must be less than the max width and height")
        if self.w < 0 or self.h < 0:
            raise ValueError("Width and height must be positive")

        # ===========FIXABLE PROBLEMS===========
        if self.x < 0:
            object.__setattr__(self, "x", 0)
        if self.y < 0:
            object.__setattr__(self, "y", 0)
        if self.x + self.w > self.max_width:
            object.__setattr__(self, "w", self.max_width - self.x)
        if self.y + self.h > self.max_height:
            object.__setattr__(self, "h", self.max_height - self.y)

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def x_center(self) -> float:
        return self.x + self.w / 2

    @property
    def y_center(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_normalized(self) -> "COCOBoundingBox":
        return COCOBoundingBox(
            self.x / self.max_width,
            self.y / self.max_height,
            self.w / self.max_width,
            self.h / self.max_height)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    def to_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def iou(self, other: "COCOBoundingBox") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union

    def shape_similarity(self, other: "COCOBoundingBox") -> float:
        intersection = min(self.w, other.w) * min(self.h, other.h)
        union = self.area + other.area - intersection
        return intersection / union


@dataclass(frozen=True)
class COCOAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    bbox: COCOBoundingBox
