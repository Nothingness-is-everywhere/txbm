import subprocess
import time
from ctypes import windll, wintypes

from PySide6.QtCore import Qt, Signal, QCoreApplication
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget, QFileDialog, QCompleter, QVBoxLayout, QHBoxLayout, QSizePolicy
from _ctypes import byref
from qfluentwidgets import PushButton, FlowLayout, ComboBox, SearchLineEdit, TextEdit, BodyLabel, StrongBodyLabel

from ok import Config, og
from ok import Handler
from ok import Logger
from ok.capture.windows.dump import dump_threads
from ok.device.capture import ImageCaptureMethod
from ok.device.interaction import DoNothingInteraction
from ok.gui.debug.RegionSelectImageWidget import RegionSelectImageWidget
from ok.gui.i18n.GettextTranslator import convert_to_mo_files
from ok.gui.util.Alert import alert_info, alert_error
from ok.gui.widget.Tab import Tab

logger = Logger.get_logger(__name__)


class DebugTab(Tab):
    update_result_text: Signal = Signal(str)

    def __init__(self, app_config, exit_event):

        super().__init__()
        self.config = Config('debug', {'target_task': "", 'target_images': [], 'target_function': ""}
                             )
        tool_widget = QWidget()
        layout = FlowLayout(tool_widget, False)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(20)
        layout.setHorizontalSpacing(10)
        self.handler = Handler(exit_event, "DebugTab")

        self.add_card(self.tr("Debug Tool"), tool_widget)

        dump_button = PushButton(self.tr("Dump Threads(HotKey:Ctrl+Alt+D)"))
        dump_button.clicked.connect(lambda: self.handler.post(dump_threads))
        layout.addWidget(dump_button)
        # self.dump_shortcut = QShortcut(QKeySequence("Ctrl+Alt+D"), self)
        # self.dump_shortcut.activated.connect(dump_threads)

        capture_button = PushButton(self.tr("Capture Screenshot"))
        capture_button.clicked.connect(lambda: self.handler.post(capture))
        layout.addWidget(capture_button)

        ocr_button = PushButton(self.tr("OCR"))
        ocr_button.clicked.connect(lambda: self.handler.post(self.ocr_log))
        layout.addWidget(ocr_button)

        gen_tr_button = PushButton(self.tr("Generate i18n files"))
        gen_tr_button.clicked.connect(self.gen_tr)
        layout.addWidget(gen_tr_button)

        convert_tr_button = PushButton(self.tr("Convert i18n files"))
        convert_tr_button.clicked.connect(convert_to_mo_files)
        layout.addWidget(convert_tr_button)

        call_task_widget = QWidget()
        call_task_container = QVBoxLayout(call_task_widget)
        call_task_layout = QHBoxLayout()
        call_task_container.addLayout(call_task_layout)
        self.add_card(self.tr("Debug Task Function"), call_task_widget)
        images = self.config.get("target_images")
        self.select_screenshot_button = PushButton(
            self.tr('{num} images selected').format(num=len(images)) if images else self.tr("Select Screenshots"))
        self.select_screenshot_button.clicked.connect(self.select_screenshot)
        call_task_layout.addWidget(self.select_screenshot_button)

        self.tasks_combo_box = ComboBox()
        call_task_layout.addWidget(self.tasks_combo_box)
        tasks = og.executor.get_all_tasks()
        class_names = [obj.__class__.__name__ for obj in tasks]
        self.tasks_combo_box.addItems(class_names)
        self.tasks_combo_box.currentTextChanged.connect(self.task_changed)

        self.target_function_edit = SearchLineEdit(self)
        self.target_function_edit.setPlaceholderText(self.tr('Type a function or property'))
        self.target_function_edit.setClearButtonEnabled(True)
        if self.config.get('target_function'):
            self.target_function_edit.setText(self.config.get('target_function'))
        call_task_layout.addWidget(self.target_function_edit, stretch=1)

        if class_names and self.config.get('target_task') in class_names:
            self.tasks_combo_box.setText(self.config.get('target_task'))
            self.task_changed(self.config.get('target_task'))
        elif class_names:
            self.tasks_combo_box.setCurrentIndex(0)
            self.config['target_task'] = class_names[0]

        self.call_button = PushButton(self.tr("Call"))
        self.call_button.clicked.connect(lambda: self.handler.post(self.call))

        call_task_layout.addWidget(self.call_button)

        self.result_edit = TextEdit()
        call_task_container.addWidget(self.result_edit, stretch=1)
        self.update_result_text.connect(self.result_edit.setText)
        self.handler.post(self.bind_hot_keys)
        self.handler.post(self.check_hotkey, 0.1)

        self._init_region_test_panel()

        og.app.app.aboutToQuit.connect(self.unregister)

    def _init_region_test_panel(self):
        """Initialize the screenshot region test panel."""
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        capture_btn = PushButton(self.tr("Capture & Load"))
        capture_btn.clicked.connect(self._region_capture_and_load)
        toolbar.addWidget(capture_btn)

        load_btn = PushButton(self.tr("Load Image"))
        load_btn.clicked.connect(self._region_load_image)
        toolbar.addWidget(load_btn)

        clear_btn = PushButton(self.tr("Clear Selection"))
        clear_btn.clicked.connect(self._region_clear_selection)
        toolbar.addWidget(clear_btn)

        toolbar.addStretch()

        self.mouse_pos_label = BodyLabel(self.tr("Mouse: --"))
        self.mouse_pos_label.setStyleSheet("color: gray;")
        toolbar.addWidget(self.mouse_pos_label)

        panel_layout.addLayout(toolbar)

        self.region_widget = RegionSelectImageWidget()
        self.region_widget.setMinimumHeight(300)
        self.region_widget.region_selected.connect(self._on_region_selected)
        self.region_widget.mouse_moved.connect(self._on_mouse_moved)
        panel_layout.addWidget(self.region_widget, stretch=1)

        result_layout = QHBoxLayout()
        result_layout.setSpacing(8)

        result_label = StrongBodyLabel(self.tr("Selection Info:"))
        result_layout.addWidget(result_label)

        self.result_edit = TextEdit()
        self.result_edit.setFixedHeight(100)
        self.result_edit.setPlaceholderText(self.tr("Select a region on the image to see coordinates..."))
        result_layout.addWidget(self.result_edit, stretch=1)

        copy_btn = PushButton(self.tr("Copy"))
        copy_btn.clicked.connect(self._copy_selection_info)
        result_layout.addWidget(copy_btn)

        panel_layout.addLayout(result_layout)

        self.add_card(self.tr("Screenshot Region Test"), panel_widget, stretch=1)

        self._last_selection_info = None

    def _region_capture_and_load(self):
        """Capture a screenshot and load it into the region widget."""
        self.handler.post(self._do_capture_and_load)

    def _do_capture_and_load(self):
        """Execute capture and load in handler thread."""
        try:
            if og.device_manager.capture_method is None:
                self.handler.post(lambda: alert_error(self.tr("No Capture Available or Selected")))
                return
            frame = og.device_manager.capture_method.get_frame()
            if frame is not None:
                self.region_widget.set_image_from_array(frame)
                self.handler.post(lambda: alert_info(self.tr("Capture loaded successfully")))
            else:
                self.handler.post(lambda: alert_error(self.tr("Capture returned None")))
        except Exception as e:
            logger.error(f"Capture and load error: {e}")
            self.handler.post(lambda: alert_error(str(e)))

    def _region_load_image(self):
        """Load an image file into the region widget."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open Image"), "",
            "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
        if file_path:
            if self.region_widget.set_image(file_path):
                alert_info(self.tr("Image loaded"))
            else:
                alert_error(self.tr("Failed to load image"))

    def _region_clear_selection(self):
        """Clear the current selection."""
        self.region_widget.clear_selection()
        self.result_edit.clear()
        self._last_selection_info = None

    def _on_region_selected(self, info: dict):
        """Handle region selection and display coordinate info."""
        self._last_selection_info = info
        img_w, img_h = self.region_widget.get_image_dimensions()

        lines = [
            self.tr("--- {title} ---").format(title=self.tr('Selection Info')),
            self.tr("x={x}, y={y}, w={w}, h={h}").format(**{k: info[k] for k in ['x', 'y', 'w', 'h']}),
            self.tr("x1={x1}, y1={y1}, x2={x2}, y2={y2}").format(**{k: info[k] for k in ['x1', 'y1', 'x2', 'y2']}),
            self.tr("ratio: rx={rx:.4f}, ry={ry:.4f}, rw={rw:.4f}, rh={rh:.4f}").format(
                rx=info['rx'], ry=info['ry'], rw=info['rw'], rh=info['rh']),
            self.tr("center: ({cx}, {cy}) = ({crx:.4f}, {cry:.4f})").format(
                cx=info['center_x'], cy=info['center_y'],
                crx=info['center_rx'], cry=info['center_ry']),
            self.tr("image size: {w}x{h}").format(w=img_w, h=img_h),
        ]
        self.result_edit.setPlainText("\n".join(lines))

    def _on_mouse_moved(self, px: int, py: int, rx: float, ry: float):
        """Update mouse position label."""
        self.mouse_pos_label.setText(
            self.tr("Mouse: ({px}, {py}) ratio: ({rx:.3f}, {ry:.3f})").format(
                px=px, py=py, rx=rx, ry=ry
            )
        )

    def _copy_selection_info(self):
        """Copy selection info to clipboard."""
        if self._last_selection_info is None:
            alert_info(self.tr("No selection to copy"))
            return

        info = self._last_selection_info
        text = self.tr(
            "x={x}, y={y}, w={w}, h={h}\n"
            "x1={x1}, y1={y1}, x2={x2}, y2={y2}\n"
            "rx={rx:.4f}, ry={ry:.4f}, rw={rw:.4f}, rh={rh:.4f}\n"
            "center: ({cx}, {cy}) ratio: ({crx:.4f}, {cry:.4f})"
        ).format(
            x=info['x'], y=info['y'], w=info['w'], h=info['h'],
            x1=info['x1'], y1=info['y1'], x2=info['x2'], y2=info['y2'],
            rx=info['rx'], ry=info['ry'], rw=info['rw'], rh=info['rh'],
            cx=info['center_x'], cy=info['center_y'],
            crx=info['center_rx'], cry=info['center_ry']
        )
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(text)
        alert_info(self.tr("Copied to clipboard"))

    def gen_tr(self):
        folder = og.app.gen_tr_po_files()
        subprocess.Popen(r'explorer /select,"{}"'.format(folder))

    def check_hotkey(self):
        # Example event type, you should use the appropriate QEvent.Type for your case
        msg = wintypes.MSG()

        # PeekMessageW is used to check for a hotkey press
        if windll.user32.PeekMessageW(byref(msg), None, 0, 0, 1):
            if msg.message == 0x0312:  # WM_HOTKEY
                logger.debug(f'hotkey pressed {msg}')
                if msg.wParam == 1:
                    logger.debug('dumping threads')
                    dump_threads()
                elif msg.wParam == 2:
                    self.handler.post(capture)

        # Repost the check_hotkey method to be called after 100 ms
        self.handler.post(self.check_hotkey, 0.1)

    def bind_hot_keys(self):
        MOD_ALT = 0x0001
        MOD_CONTROL = 0x0002
        VK_D = 0x44  # Virtual-Key code for 'D'
        VK_S = 0x53

        if not windll.user32.RegisterHotKey(None, 1, MOD_ALT | MOD_CONTROL, VK_D):
            logger.debug("Failed to register hotkey for Alt+Ctrl+D")
        if not windll.user32.RegisterHotKey(None, 2, MOD_ALT | MOD_CONTROL, VK_S):
            logger.debug("Failed to register hotkey for Alt+Ctrl+S")
        logger.debug('bind_hot_keys')

    @staticmethod
    def unregister():
        # Unregister the hotkeys
        logger.debug('Unregister the hotkeys')
        windll.user32.UnregisterHotKey(None, 1)
        windll.user32.UnregisterHotKey(None, 2)

    def call(self):
        func_name = self.target_function_edit.text()
        task_name = self.config.get('target_task')
        task = og.executor.get_task_by_class_name(task_name)

        if not hasattr(task, func_name):
            alert_error(self.tr(f"No such attr: {func_name}"))
            return
        old_capture = og.device_manager.capture_method
        old_interaction = og.device_manager.interaction
        try:
            images = self.config.get("target_images")
            if images:
                og.device_manager.capture_method = ImageCaptureMethod(og.device_manager.exit_event, images)
                og.device_manager.interaction = DoNothingInteraction(og.device_manager.capture_method)
            else:
                task.next_frame()
            og.executor.debug_mode = True
            attr = getattr(task, func_name)
            if callable(attr):
                result = str(attr())
            else:
                result = str(attr)
        except Exception as e:
            logger.error('debug call exception', e)
            result = Logger.exception_to_str(e)
        og.executor.debug_mode = False
        og.device_manager.capture_method = old_capture
        og.device_manager.interaction = old_interaction
        self.update_result_text.emit(result)
        self.config['target_function'] = func_name
        alert_info(self.tr("call success: {result}").format(result=result))

    def ocr_log(self):
        try:
            result = og.executor.get_all_tasks()[0].ocr(log=True)
            self.update_result_text.emit(str(result))
            alert_info(self.tr("OCR success (Logged): {result}").format(result=result))
            folder = og.ok.screenshot.screenshot_folder
            if folder:
                subprocess.Popen(r'explorer "{}"'.format(folder))
        except Exception as e:
            logger.error('debug ocr_log exception', e)
            return

    def task_changed(self, text):
        self.config['target_task'] = text
        task = og.executor.get_task_by_class_name(text)
        functions = [func for func in dir(task)]
        completer = QCompleter(functions, self.target_function_edit)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setMaxVisibleItems(10)
        self.target_function_edit.setCompleter(completer)

    def select_screenshot(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, self.tr("Open Image"), "", self.tr("Image Files (*.png *.jpg *.bmp)"))

        if file_names:
            logger.info(f"Selected files: {file_names}")
            self.select_screenshot_button.setText(self.tr('{num} images selected').format(num=len(file_names)))
            self.config['target_images'] = file_names
        else:
            self.select_screenshot_button.setText(self.tr("Drop or Select Screenshot"))
            self.config['target_images'] = None


def capture(processor=None):
    if og.device_manager.capture_method is not None:
        logger.info(f'og.device_manager.capture_method {og.device_manager.capture_method}')
        try:
            frame = og.device_manager.capture_method.get_frame()
            if frame is not None:
                current_capture = str(og.device_manager.capture_method) + '_' + str(time.time() * 1000)
                file_path = og.ok.screenshot.generate_screen_shot(frame, og.ok.screenshot.ui_dict,
                                                                  og.ok.screenshot.screenshot_folder,
                                                                  current_capture, True, None, processor=processor)

                # Use subprocess.Popen to open the file explorer and select the file
                subprocess.Popen(r'explorer /select,"{}"'.format(file_path))
                logger.info(f'captured screenshot: {current_capture}')
                alert_info(QCoreApplication.translate('DebugTab', 'Capture Success'), True)
            else:
                alert_error(QCoreApplication.translate('DebugTab', 'Capture returned None'))
        except Exception as e:
            alert_error(QCoreApplication.translate('DebugTab', str(e)))
    else:
        alert_error(QCoreApplication.translate('DebugTab', 'No Capture Available or Selected'))
