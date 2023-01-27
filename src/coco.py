from dataclasses import dataclass


@dataclass(frozen=True)
class COCOBoundingBox:
    x: float # Top left corner
    y: float # Top left corner
    w: float # Width
    h: float # Height

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

    def to_normalized(self, max_width: int, max_height: int) -> "COCOBoundingBox":
        return COCOBoundingBox(
            self.x / max_width,
            self.y / max_height,
            self.w / max_width,
            self.h / max_height)

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
