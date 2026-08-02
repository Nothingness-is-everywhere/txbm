"""Region-selectable image widget for screenshot testing.

Provides a reusable component that displays an image and allows the user
to drag-select a rectangular region, emitting coordinate information in
multiple formats (x,y,w,h / x1,y1,x2,y2 / ratio).

Reuses DPI/scaling patterns from OverlayWidget.
"""

from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PySide6.QtWidgets import QWidget, QSizePolicy

from ok import Logger

logger = Logger.get_logger(__name__)


class RegionSelectImageWidget(QWidget):
    """A widget that displays an image and supports drag-to-select a region.

    Signals:
        region_selected: Emitted when a region is selected with a dict containing:
            - x, y, w, h: absolute pixel coordinates
            - x1, y1, x2, y2: corner coordinates
            - rx, ry, rw, rh: ratio coordinates (0.0-1.0)
            - center_x, center_y: center pixel coordinates
            - center_rx, center_ry: center ratio coordinates
    """

    region_selected = Signal(dict)
    mouse_moved = Signal(int, int, float, float)  # px, py, ratio_x, ratio_y

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._display_pixmap: Optional[QPixmap] = None
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

        self._dragging: bool = False
        self._start_pos: QPoint = QPoint()
        self._end_pos: QPoint = QPoint()
        self._selection: Optional[QRect] = None

        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_image(self, image_path: str) -> bool:
        """Load and display an image from file."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            logger.warning(f"Failed to load image: {image_path}")
            return False
        self._pixmap = pixmap
        self._update_display_pixmap()
        self._selection = None
        self.update()
        return True

    def set_image_from_array(self, image_array) -> bool:
        """Load image from numpy array (BGR format)."""
        try:
            import cv2
            from PIL import Image as PILImage
            import io
            if len(image_array.shape) == 2:
                rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height,
                            QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimage)
            self._update_display_pixmap()
            self._selection = None
            self.update()
            return True
        except Exception as e:
            logger.error(f"Failed to load image from array: {e}")
            return False

    def set_image_from_qimage(self, qimage: QImage):
        """Set image directly from QImage."""
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_display_pixmap()
        self._selection = None
        self.update()

    def _update_display_pixmap(self):
        """Scale the pixmap to fit the widget while maintaining aspect ratio."""
        if self._pixmap is None:
            return
        # Widget has no valid size yet (e.g. before being shown in a dialog);
        # skip until resizeEvent fires with a real geometry to avoid divide-by-zero.
        if self.width() <= 0 or self.height() <= 0:
            return

        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self._display_pixmap = scaled

        self._offset_x = (self.width() - scaled.width()) // 2
        self._offset_y = (self.height() - scaled.height()) // 2

        if scaled.width() > 0 and scaled.height() > 0:
            self._scale_x = float(self._pixmap.width()) / float(scaled.width())
            self._scale_y = float(self._pixmap.height()) / float(scaled.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display_pixmap()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._display_pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, self.tr("No image loaded"))
            return

        painter.drawPixmap(self._offset_x, self._offset_y, self._display_pixmap)

        if self._selection is not None:
            pen = QPen(QColor(255, 0, 0, 200), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 0, 0, 50))
            painter.drawRect(self._selection)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._display_pixmap is not None:
            img_pos = self._widget_to_image(event.pos())
            if img_pos:
                self._dragging = True
                self._start_pos = self._image_to_widget(img_pos)
                self._end_pos = self._start_pos
                self._selection = QRect(self._start_pos, self._end_pos)
                self.update()

    def mouseMoveEvent(self, event):
        img_pos = self._widget_to_image(event.pos())
        if img_pos and self._pixmap:
            ratio_x = float(img_pos.x()) / self._pixmap.width() if self._pixmap.width() > 0 else 0
            ratio_y = float(img_pos.y()) / self._pixmap.height() if self._pixmap.height() > 0 else 0
            self.mouse_moved.emit(img_pos.x(), img_pos.y(), ratio_x, ratio_y)

        if self._dragging and self._display_pixmap is not None:
            img_pos = self._widget_to_image(event.pos())
            if img_pos:
                self._end_pos = self._image_to_widget(img_pos)
                self._selection = QRect(self._start_pos, self._end_pos).normalized()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if self._selection is not None and self._selection.width() > 2 and self._selection.height() > 2:
                self._emit_selection()

    def _widget_to_image(self, pos: QPoint) -> Optional[QPoint]:
        """Convert widget coordinates to original image coordinates."""
        if self._display_pixmap is None or self._pixmap is None:
            return None

        rel_x = pos.x() - self._offset_x
        rel_y = pos.y() - self._offset_y

        if rel_x < 0 or rel_y < 0 or rel_x > self._display_pixmap.width() or rel_y > self._display_pixmap.height():
            return None

        img_x = int(rel_x * self._scale_x)
        img_y = int(rel_y * self._scale_y)
        return QPoint(img_x, img_y)

    def _image_to_widget(self, img_pos: QPoint) -> QPoint:
        """Convert original image coordinates to widget coordinates."""
        wx = int(img_pos.x() / self._scale_x) + self._offset_x
        wy = int(img_pos.y() / self._scale_y) + self._offset_y
        return QPoint(wx, wy)

    def _emit_selection(self):
        """Emit the selected region with coordinate information."""
        if self._selection is None or self._pixmap is None:
            return

        tl = self._widget_to_image(self._selection.topLeft())
        br = self._widget_to_image(self._selection.bottomRight())

        if tl is None or br is None:
            return

        x, y = tl.x(), tl.y()
        x2, y2 = br.x(), br.y()
        w = x2 - x
        h = y2 - y

        img_w = self._pixmap.width()
        img_h = self._pixmap.height()

        rx = float(x) / img_w if img_w > 0 else 0
        ry = float(y) / img_h if img_h > 0 else 0
        rw = float(w) / img_w if img_w > 0 else 0
        rh = float(h) / img_h if img_h > 0 else 0

        cx = x + w // 2
        cy = y + h // 2
        crx = float(cx) / img_w if img_w > 0 else 0
        cry = float(cy) / img_h if img_h > 0 else 0

        self.region_selected.emit({
            "x": x, "y": y, "w": w, "h": h,
            "x1": x, "y1": y, "x2": x2, "y2": y2,
            "rx": rx, "ry": ry, "rw": rw, "rh": rh,
            "center_x": cx, "center_y": cy,
            "center_rx": crx, "center_ry": cry,
            "image_w": img_w, "image_h": img_h,
        })

    def get_current_selection_info(self) -> Optional[dict]:
        """Get current selection info without requiring a new selection."""
        if self._selection is None:
            return None
        result = {}
        self._emit_selection()
        return result

    def clear_selection(self):
        """Clear the current selection."""
        self._selection = None
        self.update()

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Return the original image dimensions."""
        if self._pixmap:
            return (self._pixmap.width(), self._pixmap.height())
        return (0, 0)

    def get_pixmap(self) -> Optional[QPixmap]:
        """Return the current pixmap for reuse (e.g. in a maximized view)."""
        return self._pixmap

    def get_cropped_pixmap(self) -> Optional[QPixmap]:
        """Return a deep copy of the current selection in original image coordinates.

        Returns None if no image is loaded or no valid selection exists.
        The returned pixmap is clipped to the image bounds.
        """
        if self._pixmap is None or self._selection is None:
            return None
        tl = self._widget_to_image(self._selection.topLeft())
        br = self._widget_to_image(self._selection.bottomRight())
        if tl is None or br is None:
            return None
        x, y = tl.x(), tl.y()
        x2, y2 = br.x(), br.y()
        if x2 <= x or y2 <= y:
            return None
        rect = QRect(x, y, x2 - x, y2 - y).intersected(self._pixmap.rect())
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        return self._pixmap.copy(rect)

    def set_pixmap(self, pixmap: QPixmap):
        """Set image directly from a QPixmap (reuses without conversion)."""
        self._pixmap = pixmap
        self._update_display_pixmap()
        self._selection = None
        self.update()

    def set_selection_from_image(self, x: int, y: int, w: int, h: int):
        """Set the selection rect from original image coordinates."""
        if self._pixmap is None or self._display_pixmap is None:
            return
        tl = self._image_to_widget(QPoint(x, y))
        br = self._image_to_widget(QPoint(x + w, y + h))
        self._selection = QRect(tl, br).normalized()
        self.update()