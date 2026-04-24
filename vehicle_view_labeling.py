import os
import sys
import cv2
import glob
import re
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import YOLO

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
DET_MODEL_PATH = r"F:\18744\runs\detect\runs_detect\ft_bestpt_640\weights\best.pt"
JPEG_QUALITY = 100
DET_CONF = 0.45
TRUST_CONF = 0.65
IOU_THRES = 0.7
DEVICE_ID = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================
# UTILS
# ==========================================
def clip_bbox_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_images(image_dir):
    image_paths = []
    for name in os.listdir(image_dir):
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_paths.append(path)
    image_paths.sort(key=lambda p: natural_key(os.path.basename(p)))
    return image_paths


# ==========================================
# DETECTION THREAD
# ==========================================
class DetectionThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, int)
    finished_detection = QtCore.pyqtSignal(str)

    def __init__(self, image_dir, output_dir):
        super().__init__()
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        dataset_name = os.path.basename(os.path.normpath(self.image_dir))
        crops_root = os.path.join(self.output_dir, dataset_name)
        os.makedirs(crops_root, exist_ok=True)

        try:
            model = YOLO(DET_MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        image_paths = list_images(self.image_dir)
        total_images = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            if not self.is_running:
                break

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_out_dir = os.path.join(crops_root, image_name)
            os.makedirs(image_out_dir, exist_ok=True)

            frame = cv2.imread(image_path)
            if frame is None:
                self.progress_update.emit(idx + 1, total_images)
                continue

            h, w = frame.shape[:2]
            try:
                results = model.predict(
                    source=frame,
                    conf=DET_CONF,
                    iou=IOU_THRES,
                    device=DEVICE_ID,
                    verbose=False
                )[0]
            except Exception as e:
                print(f"Detection failed on {image_path}: {e}")
                self.progress_update.emit(idx + 1, total_images)
                continue

            if results.boxes is not None and results.boxes.xyxy is not None:
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None

                for det_idx in range(len(xyxy)):
                    conf = float(confs[det_idx]) if confs is not None else 1.0
                    if conf < TRUST_CONF:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy[det_idx])
                    x1, y1, x2, y2 = clip_bbox_xyxy(x1, y1, x2, y2, w, h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    out_path = os.path.join(
                        image_out_dir,
                        f"{image_name}_det{det_idx:06d}.jpg"
                    )
                    cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            self.progress_update.emit(idx + 1, total_images)

        self.finished_detection.emit(crops_root)

    def stop(self):
        self.is_running = False


# ==========================================
# DATA MODEL
# ==========================================
class ImageData:
    def __init__(self, image_name, folder):
        self.image_name = image_name
        self.folder = folder
        self.crops = []
        self.crop_labels = {}

    def load_existing_labels(self):
        summary_path = os.path.join(self.folder, "labels_image.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        crop_idx = int(parts[0])
                        view_label = parts[1]
                        self.crop_labels[crop_idx] = {"view": view_label}

        for crop_idx, crop_path in self.crops:
            txt_path = os.path.splitext(crop_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    parts = f.read().strip().split()
                    if len(parts) >= 2:
                        self.crop_labels[crop_idx] = {"view": parts[1]}

    def save_summary_labels(self):
        summary_path = os.path.join(self.folder, "labels_image.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            for crop_idx in sorted(self.crop_labels.keys()):
                f.write(f"{crop_idx} {self.crop_labels[crop_idx]['view']}\n")

    def save_crop_labels(self):
        for crop_idx, label in self.crop_labels.items():
            crop_path = next((path for idx, path in self.crops if idx == crop_idx), None)
            if crop_path:
                txt_path = os.path.splitext(crop_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"{crop_idx} {label['view']}\n")
        self.save_summary_labels()


# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
class LabelingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Vehicle View Labeler")
        self.resize(1100, 750)

        self.images = []
        self.current_image_idx = 0
        self.current_crop_idx = 0
        self.det_thread = None

        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        top_bar = QtWidgets.QHBoxLayout()
        btn_load_dir = QtWidgets.QPushButton("1. Select Image Directory & Detect")
        btn_load_dir.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_load_dir.clicked.connect(self.select_and_detect_directory)

        self.lbl_status = QtWidgets.QLabel("Status: Waiting for image directory...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        top_bar.addWidget(btn_load_dir)
        top_bar.addWidget(self.lbl_status)
        top_bar.addWidget(self.progress_bar, stretch=1)
        main_layout.addLayout(top_bar)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(line)

        split_layout = QtWidgets.QHBoxLayout()

        left_panel = QtWidgets.QVBoxLayout()
        nav_layout = QtWidgets.QHBoxLayout()
        self.lbl_progress = QtWidgets.QLabel("Image 0 / 0")
        btn_prev_img = QtWidgets.QPushButton("<< Prev Image")
        btn_prev_img.clicked.connect(lambda: self.load_image_group(self.current_image_idx - 1))
        btn_next_img = QtWidgets.QPushButton("Next Image >>")
        btn_next_img.clicked.connect(lambda: self.load_image_group(self.current_image_idx + 1))
        nav_layout.addWidget(self.lbl_progress)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_prev_img)
        nav_layout.addWidget(btn_next_img)
        left_panel.addLayout(nav_layout)

        self.image_label = QtWidgets.QLabel("Select an image directory above to begin.")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        left_panel.addWidget(self.image_label, stretch=1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider_move)
        left_panel.addWidget(self.slider)

        ctrl_layout = QtWidgets.QHBoxLayout()
        btn_step_b = QtWidgets.QPushButton("< Prev Crop")
        btn_step_b.clicked.connect(lambda: self.step_crop(-1))
        btn_step_f = QtWidgets.QPushButton("Next Crop >")
        btn_step_f.clicked.connect(lambda: self.step_crop(1))

        self.lbl_crop_info = QtWidgets.QLabel("Crop: 0 / 0")
        ctrl_layout.addWidget(btn_step_b)
        ctrl_layout.addWidget(btn_step_f)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_crop_info)
        left_panel.addLayout(ctrl_layout)

        self.lbl_image_name = QtWidgets.QLabel("Source Image: -")
        left_panel.addWidget(self.lbl_image_name)

        split_layout.addLayout(left_panel, stretch=2)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(QtWidgets.QLabel("<b>Vehicle View</b>"))
        self.view_group = QtWidgets.QButtonGroup()
        for txt in ["front", "back", "unknown"]:
            rb = QtWidgets.QRadioButton(txt)
            self.view_group.addButton(rb)
            right_panel.addWidget(rb)
        self.view_group.buttons()[0].setChecked(True)

        btn_save_crop = QtWidgets.QPushButton("Save File & Next >")
        btn_save_crop.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; font-weight: bold;")
        btn_save_crop.clicked.connect(self.save_current_and_next)

        btn_copy_prev = QtWidgets.QPushButton("Copy Previous Crop Label")
        btn_copy_prev.clicked.connect(self.copy_prev_crop_label)

        right_panel.addWidget(btn_save_crop)
        right_panel.addWidget(btn_copy_prev)

        right_panel.addSpacing(20)
        right_panel.addWidget(QtWidgets.QLabel("<b>Saved Crop Labels: (Double-click to jump)</b>"))
        self.list_crop_labels = QtWidgets.QListWidget()
        self.list_crop_labels.itemDoubleClicked.connect(self.jump_to_crop_from_list)
        right_panel.addWidget(self.list_crop_labels)

        btn_del_crop_label = QtWidgets.QPushButton("Delete Selected Crop Label")
        btn_del_crop_label.clicked.connect(self.delete_selected_crop_label)
        right_panel.addWidget(btn_del_crop_label)

        split_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(split_layout, stretch=1)

    # ==========================================
    # DETECTION WORKFLOW
    # ==========================================
    def select_and_detect_directory(self):
        image_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not image_dir:
            return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory for Crops/Labels")
        if not output_dir:
            return

        self.lbl_status.setText(f"Detecting vehicles in: {os.path.basename(image_dir)}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.det_thread = DetectionThread(image_dir, output_dir)
        self.det_thread.progress_update.connect(self.update_progress)
        self.det_thread.finished_detection.connect(self.on_detection_finished)
        self.det_thread.start()

    def update_progress(self, current, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_detection_finished(self, crops_root):
        self.lbl_status.setText("Detection complete! Loading crops...")
        self.progress_bar.setVisible(False)
        self.load_data(crops_root)

    def load_data(self, crops_root):
        self.images = []
        folders = [f for f in glob.glob(os.path.join(crops_root, "*")) if os.path.isdir(f)]
        folders.sort(key=lambda p: natural_key(os.path.basename(p)))

        for folder in folders:
            image_name = os.path.basename(folder)
            data = ImageData(image_name, folder)
            crops = glob.glob(os.path.join(folder, "*.jpg"))
            crops.sort(key=natural_key)

            for crop_path in crops:
                m = re.search(r"_det(\d+)\.", os.path.basename(crop_path))
                if m:
                    data.crops.append((int(m.group(1)), crop_path))

            if data.crops:
                data.load_existing_labels()
                self.images.append(data)

        if self.images:
            self.lbl_status.setText("Ready for labeling.")
            self.load_image_group(0)
        else:
            self.lbl_status.setText("No detections found in this directory.")
            self.image_label.setText("No detections found.")
            self.lbl_progress.setText("Image 0 / 0")
            self.lbl_crop_info.setText("Crop: 0 / 0")
            self.lbl_image_name.setText("Source Image: -")
            self.list_crop_labels.clear()

    # ==========================================
    # DISPLAY & LABEL LOGIC
    # ==========================================
    def load_image_group(self, idx):
        if not self.images or idx < 0 or idx >= len(self.images):
            return
        self.current_image_idx = idx
        self.current_crop_idx = 0
        current = self.images[self.current_image_idx]

        self.lbl_progress.setText(f"Image {idx + 1} / {len(self.images)}")
        self.lbl_image_name.setText(f"Source Image: {current.image_name}")
        self.slider.setRange(0, len(current.crops) - 1)
        self.refresh_crop_label_list()
        self.update_display()

    def update_display(self):
        if not self.images:
            return

        current = self.images[self.current_image_idx]
        crop_idx, crop_path = current.crops[self.current_crop_idx]

        pixmap = QtGui.QPixmap(crop_path)
        pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        self.lbl_crop_info.setText(f"Crop: {self.current_crop_idx + 1} / {len(current.crops)} | Det ID: {crop_idx:06d}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_crop_idx)
        self.slider.blockSignals(False)

        label = current.crop_labels.get(crop_idx)
        if label:
            for btn in self.view_group.buttons():
                if btn.text() == label["view"]:
                    btn.setChecked(True)
                    break
        else:
            self.view_group.buttons()[0].setChecked(True)

    def on_slider_move(self, val):
        self.current_crop_idx = val
        self.update_display()

    def step_crop(self, direction):
        if not self.images:
            return
        current = self.images[self.current_image_idx]
        self.current_crop_idx += direction
        self.current_crop_idx = max(0, min(self.current_crop_idx, len(current.crops) - 1))
        self.update_display()

    def save_current_and_next(self):
        current = self.images[self.current_image_idx]
        crop_idx = current.crops[self.current_crop_idx][0]
        current.crop_labels[crop_idx] = {
            "view": self.view_group.checkedButton().text()
        }
        current.save_crop_labels()
        self.refresh_crop_label_list()

        if self.current_crop_idx < len(current.crops) - 1:
            self.current_crop_idx += 1
            self.update_display()
        elif self.current_image_idx < len(self.images) - 1:
            self.load_image_group(self.current_image_idx + 1)

    def copy_prev_crop_label(self):
        current = self.images[self.current_image_idx]
        crop_idx = current.crops[self.current_crop_idx][0]
        prev_keys = [k for k in current.crop_labels.keys() if k < crop_idx]
        if not prev_keys:
            return

        last_label = current.crop_labels[max(prev_keys)]
        for btn in self.view_group.buttons():
            if btn.text() == last_label["view"]:
                btn.setChecked(True)
                break
        self.save_current_and_next()

    def refresh_crop_label_list(self):
        self.list_crop_labels.clear()
        if not self.images:
            return

        current = self.images[self.current_image_idx]
        for crop_idx in sorted(current.crop_labels.keys()):
            label = current.crop_labels[crop_idx]
            item_text = f"Det {crop_idx:06d}: {label['view']}"
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, crop_idx)
            self.list_crop_labels.addItem(item)

    def jump_to_crop_from_list(self, item):
        target_crop_idx = item.data(QtCore.Qt.UserRole)
        current = self.images[self.current_image_idx]
        for internal_idx, crop_data in enumerate(current.crops):
            if crop_data[0] == target_crop_idx:
                self.current_crop_idx = internal_idx
                self.update_display()
                break

    def delete_selected_crop_label(self):
        row = self.list_crop_labels.currentRow()
        if row < 0:
            return

        item = self.list_crop_labels.item(row)
        target_crop_idx = item.data(QtCore.Qt.UserRole)
        current = self.images[self.current_image_idx]

        if target_crop_idx in current.crop_labels:
            del current.crop_labels[target_crop_idx]

            crop_path = next((path for idx, path in current.crops if idx == target_crop_idx), None)
            if crop_path:
                txt_path = os.path.splitext(crop_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    os.remove(txt_path)

            current.save_summary_labels()

        self.refresh_crop_label_list()
        self.update_display()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())
