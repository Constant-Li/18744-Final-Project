import os
import sys
import cv2
import glob
import re
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import YOLO

# ==========================================
# GLOBAL CONFIGURATION (TUNE YOUR PARAMETERS HERE)
# ==========================================
DET_MODEL_PATH = r"F:\18744\runs\detect\runs_detect\ft_bestpt_640\weights\best.pt"
TRACKER_CFG = "botsort_reid_vehicle.yaml" 
FPS_FALLBACK = 30.0

# ---- Crop saving ----
CROP_STRIDE = 10       # Originally CROP_EVERY: Save a crop and label every N frames
JPEG_QUALITY = 100    # 0-100 (higher is better quality but larger files)

# ---- Thresholds & Tracking ----
DET_CONF = 0.40       # Permissive confidence for the tracker to maintain IDs
TRUST_CONF = 0.60     # Only crop and save if confidence is above this
IOU_THRES = 0.7       # NMS IOU threshold
DEVICE_ID = 0         # GPU device ID

# ==========================================
# TRACKING UTILS & THREAD
# ==========================================
def safe_int(x, default=None):
    try: return int(x)
    except Exception: return default

def clip_bbox_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2

class TrackerThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, int) 
    finished_tracking = QtCore.pyqtSignal(str, float) 

    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        crops_root = os.path.join(self.output_dir, video_name)
        os.makedirs(crops_root, exist_ok=True)

        try: model = YOLO(DET_MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): return

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = -1
        while self.is_running:
            ok, frame = cap.read()
            if not ok or frame is None: break
            frame_idx += 1

            if frame_idx % 10 == 0:
                self.progress_update.emit(frame_idx, total_frames)

            # Track every frame using your tuned parameters
            results = model.track(
                source=frame, 
                persist=True, 
                tracker=TRACKER_CFG,
                conf=DET_CONF, 
                iou=IOU_THRES, 
                device=DEVICE_ID, 
                verbose=False
            )[0]

            if frame_idx % CROP_STRIDE == 0:
                if results.boxes is not None and results.boxes.xyxy is not None:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None
                    ids = results.boxes.id.cpu().numpy() if getattr(results.boxes, "id", None) is not None else None

                    for i in range(len(xyxy)):
                        tid = safe_int(ids[i]) if ids is not None else None
                        if tid is None: continue
                        
                        conf = float(confs[i]) if confs is not None else 1.0
                        if conf < TRUST_CONF: continue 

                        x1, y1, x2, y2 = map(int, xyxy[i])
                        x1, y1, x2, y2 = clip_bbox_xyxy(x1, y1, x2, y2, w, h)
                        if x2 <= x1 or y2 <= y1: continue

                        crop = frame[y1:y2, x1:x2]
                        id_dir = os.path.join(crops_root, f"v_id_{tid}")
                        os.makedirs(id_dir, exist_ok=True)
                        out_path = os.path.join(id_dir, f"v_id{tid}_frame{frame_idx:06d}.jpg")
                        cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        cap.release()
        self.progress_update.emit(total_frames, total_frames)
        self.finished_tracking.emit(crops_root, fps)

    def stop(self):
        self.is_running = False

# ==========================================
# DATA MODELS
# ==========================================
class VehicleData:
    def __init__(self, vid, folder, fps):
        self.vid = vid
        self.folder = folder
        self.fps = fps
        self.frames = [] 
        self.video_intervals = [] 
        self.frame_labels = {} 
        
    def load_existing_labels(self):
        vid_path = os.path.join(self.folder, "labels_video.txt")
        if os.path.exists(vid_path):
            with open(vid_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        self.video_intervals.append({'type': parts[0], 'start': float(parts[1]), 'end': float(parts[2])})
        
        for f_idx, t_sec, img_path in self.frames:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    parts = f.read().strip().split()
                    if len(parts) >= 4:
                        self.frame_labels[f_idx] = {'turn': parts[2], 'brake': parts[3]}

    def save_video_labels(self):
        with open(os.path.join(self.folder, "labels_video.txt"), 'w') as f:
            for interval in self.video_intervals:
                f.write(f"{interval['type']} {interval['start']:.2f} {interval['end']:.2f}\n")

    def save_frame_labels(self):
        for f_idx, lbl in self.frame_labels.items():
            img_path = next((path for f, t, path in self.frames if f == f_idx), None)
            if img_path:
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                time_sec = f_idx / self.fps
                with open(txt_path, 'w') as f:
                    f.write(f"{f_idx} {time_sec:.2f} {lbl['turn']} {lbl['brake']}\n")

# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
class LabelingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Unified Labeler (Stride: {CROP_STRIDE})")
        self.resize(1100, 750)
        
        self.vehicles = []
        self.current_v_idx = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.fps = FPS_FALLBACK
        self.tracker_thread = None
        
        self.temp_turn_start = None
        self.temp_brake_start = None

        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # ---- TOP BAR ----
        top_bar = QtWidgets.QHBoxLayout()
        btn_load_video = QtWidgets.QPushButton("1. Select & Track Video")
        btn_load_video.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_load_video.clicked.connect(self.select_and_track_video)
        
        self.lbl_status = QtWidgets.QLabel("Status: Waiting for video...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        top_bar.addWidget(btn_load_video)
        top_bar.addWidget(self.lbl_status)
        top_bar.addWidget(self.progress_bar, stretch=1)
        main_layout.addLayout(top_bar)
        
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(line)

        # ---- MAIN SPLIT ----
        split_layout = QtWidgets.QHBoxLayout()

        # LEFT PANEL
        left_panel = QtWidgets.QVBoxLayout()
        nav_layout = QtWidgets.QHBoxLayout()
        self.lbl_progress = QtWidgets.QLabel("Vehicle 0 / 0")
        btn_prev_veh = QtWidgets.QPushButton("<< Prev Vehicle")
        btn_prev_veh.clicked.connect(lambda: self.load_vehicle(self.current_v_idx - 1))
        btn_next_veh = QtWidgets.QPushButton("Next Vehicle >>")
        btn_next_veh.clicked.connect(lambda: self.load_vehicle(self.current_v_idx + 1))
        nav_layout.addWidget(self.lbl_progress)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_prev_veh)
        nav_layout.addWidget(btn_next_veh)
        left_panel.addLayout(nav_layout)

        self.image_label = QtWidgets.QLabel("Select a video above to begin.")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        left_panel.addWidget(self.image_label, stretch=1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider_move)
        left_panel.addWidget(self.slider)

        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play / Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_step_b = QtWidgets.QPushButton("< Step Back")
        btn_step_b.clicked.connect(lambda: self.step_frame(-1))
        btn_step_f = QtWidgets.QPushButton("Step Forward >")
        btn_step_f.clicked.connect(lambda: self.step_frame(1))
        
        self.lbl_time = QtWidgets.QLabel("Time: 0.00s | Frame: 0")
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(btn_step_b)
        ctrl_layout.addWidget(btn_step_f)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_time)
        left_panel.addLayout(ctrl_layout)

        split_layout.addLayout(left_panel, stretch=2)

        # RIGHT PANEL: Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(380)
        
        # TAB A: Video-wise
        tab_a = QtWidgets.QWidget()
        layout_a = QtWidgets.QVBoxLayout(tab_a)
        layout_a.addWidget(QtWidgets.QLabel("<b>Turn Event</b>"))
        self.turn_group = QtWidgets.QButtonGroup()
        for txt in ["turn_left", "turn_right", "turn_hazard", "off"]:
            rb = QtWidgets.QRadioButton(txt)
            self.turn_group.addButton(rb)
            layout_a.addWidget(rb)
        self.turn_group.buttons()[-1].setChecked(True)
        
        btn_turn_start = QtWidgets.QPushButton("Set Turn Start")
        btn_turn_start.clicked.connect(lambda: self.set_interval_start('turn'))
        btn_turn_end = QtWidgets.QPushButton("Save Turn Interval (End Here)")
        btn_turn_end.clicked.connect(lambda: self.save_interval('turn'))
        layout_a.addWidget(btn_turn_start)
        layout_a.addWidget(btn_turn_end)
        
        layout_a.addSpacing(20)
        layout_a.addWidget(QtWidgets.QLabel("<b>Brake Event</b>"))
        btn_brake_start = QtWidgets.QPushButton("Set Brake Start")
        btn_brake_start.clicked.connect(lambda: self.set_interval_start('brake'))
        btn_brake_end = QtWidgets.QPushButton("Save Brake Interval (End Here)")
        btn_brake_end.clicked.connect(lambda: self.save_interval('brake'))
        layout_a.addWidget(btn_brake_start)
        layout_a.addWidget(btn_brake_end)

        layout_a.addSpacing(20)
        self.list_intervals = QtWidgets.QListWidget()
        btn_del_interval = QtWidgets.QPushButton("Delete Selected Interval")
        btn_del_interval.clicked.connect(self.delete_selected_interval)
        layout_a.addWidget(QtWidgets.QLabel("<b>Saved Intervals:</b>"))
        layout_a.addWidget(self.list_intervals)
        layout_a.addWidget(btn_del_interval)
        self.tabs.addTab(tab_a, "Step A: Video-wise")

        # TAB B: Frame-wise
        tab_b = QtWidgets.QWidget()
        layout_b = QtWidgets.QVBoxLayout(tab_b)
        
        layout_b.addSpacing(10)
        layout_b.addWidget(QtWidgets.QLabel("<b>Frame Turn State</b>"))
        self.f_turn_group = QtWidgets.QButtonGroup()
        for txt in ["off", "left", "right", "hazard"]:
            rb = QtWidgets.QRadioButton(txt)
            self.f_turn_group.addButton(rb)
            layout_b.addWidget(rb)
        self.f_turn_group.buttons()[0].setChecked(True)

        layout_b.addSpacing(10)
        layout_b.addWidget(QtWidgets.QLabel("<b>Frame Brake State</b>"))
        self.f_brake_group = QtWidgets.QButtonGroup()
        for txt in ["no_brake", "brake"]:
            rb = QtWidgets.QRadioButton(txt)
            self.f_brake_group.addButton(rb)
            layout_b.addWidget(rb)
        self.f_brake_group.buttons()[0].setChecked(True)

        btn_save_frame = QtWidgets.QPushButton("Save File & Next >")
        btn_save_frame.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; font-weight: bold;")
        btn_save_frame.clicked.connect(self.save_current_and_next)
        
        btn_copy_prev = QtWidgets.QPushButton("Copy Previous Frame Label")
        btn_copy_prev.clicked.connect(self.copy_prev_frame_label)
        
        layout_b.addWidget(btn_save_frame)
        layout_b.addWidget(btn_copy_prev)
        
        layout_b.addSpacing(20)
        layout_b.addWidget(QtWidgets.QLabel("<b>Saved Frame Labels: (Double-click to jump)</b>"))
        self.list_frame_labels = QtWidgets.QListWidget()
        self.list_frame_labels.itemDoubleClicked.connect(self.jump_to_frame_from_list)
        layout_b.addWidget(self.list_frame_labels)

        btn_del_frame_label = QtWidgets.QPushButton("Delete Selected Frame Label")
        btn_del_frame_label.clicked.connect(self.delete_selected_frame_label)
        layout_b.addWidget(btn_del_frame_label)

        self.tabs.addTab(tab_b, "Step B: Frame-wise")

        split_layout.addWidget(self.tabs, stretch=1)
        main_layout.addLayout(split_layout, stretch=1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.step_frame(1, loop=True))

    # ==========================================
    # WORKFLOW INTEGRATION
    # ==========================================
    def select_and_track_video(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if not video_path: return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory for Project/Crops")
        if not output_dir: return

        self.lbl_status.setText(f"Tracking: {os.path.basename(video_path)}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.tracker_thread = TrackerThread(video_path, output_dir)
        self.tracker_thread.progress_update.connect(self.update_progress)
        self.tracker_thread.finished_tracking.connect(self.on_tracking_finished)
        self.tracker_thread.start()

    def update_progress(self, current, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_tracking_finished(self, crops_root, fps):
        self.fps = fps
        self.lbl_status.setText("Tracking complete! Loading crops...")
        self.progress_bar.setVisible(False)
        self.load_data(crops_root)

    def load_data(self, crops_root):
        self.vehicles = []
        folders = [f for f in glob.glob(os.path.join(crops_root, "v_id_*")) if os.path.isdir(f)]
        
        for folder in folders:
            m = re.search(r'v_id_?(\d+)', os.path.basename(folder))
            if not m: continue
            vid = int(m.group(1))
            
            v_data = VehicleData(vid, folder, self.fps)
            images = glob.glob(os.path.join(folder, "*.jpg"))
            
            for img in images:
                fm = re.search(r'frame(\d+)\.', os.path.basename(img))
                if fm: v_data.frames.append((int(fm.group(1)), int(fm.group(1)) / self.fps, img))
            
            if v_data.frames:
                v_data.frames.sort(key=lambda x: x[0])
                v_data.load_existing_labels()
                self.vehicles.append(v_data)

        self.vehicles.sort(key=lambda v: v.vid)
        if self.vehicles:
            self.lbl_status.setText("Ready for labeling.")
            self.load_vehicle(0)
        else:
            self.lbl_status.setText("No vehicles tracked in this video.")

    # ==========================================
    # DISPLAY AND LOGIC
    # ==========================================
    def load_vehicle(self, idx):
        if not self.vehicles or idx < 0 or idx >= len(self.vehicles): return
        self.current_v_idx = idx
        self.current_frame_idx = 0
        v = self.vehicles[self.current_v_idx]
        
        self.lbl_progress.setText(f"Vehicle {idx + 1} / {len(self.vehicles)} (ID: {v.vid})")
        self.slider.setRange(0, len(v.frames) - 1)
        self.refresh_interval_list()
        self.refresh_frame_label_list()
        self.update_display()

    def update_display(self):
        if not self.vehicles: return
        v = self.vehicles[self.current_v_idx]
        f_idx_real, t_sec, img_path = v.frames[self.current_frame_idx]
        
        pixmap = QtGui.QPixmap(img_path)
        pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        
        self.lbl_time.setText(f"Time: {t_sec:.2f}s | Frame: {f_idx_real}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)

        lbl = v.frame_labels.get(f_idx_real)
        if lbl:
            for btn in self.f_turn_group.buttons():
                if btn.text() == lbl['turn']: btn.setChecked(True)
            for btn in self.f_brake_group.buttons():
                if btn.text() == lbl['brake']: btn.setChecked(True)
        else:
            self.f_turn_group.buttons()[0].setChecked(True)
            self.f_brake_group.buttons()[0].setChecked(True)

    def on_slider_move(self, val):
        self.current_frame_idx = val
        self.update_display()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            effective_fps = self.fps / CROP_STRIDE
            self.timer.start(int(1000 / effective_fps))
        else:
            self.btn_play.setText("Play / Pause")
            self.timer.stop()

    def step_frame(self, direction, loop=False):
        if not self.vehicles: return
        v = self.vehicles[self.current_v_idx]
        self.current_frame_idx += direction
        
        if self.current_frame_idx >= len(v.frames):
            if loop: self.current_frame_idx = 0
            else: 
                self.current_frame_idx = len(v.frames) - 1
                if self.is_playing: self.toggle_play()
        elif self.current_frame_idx < 0:
            self.current_frame_idx = 0
        self.update_display()

    # --- Video-wise Methods ---
    def get_current_time(self):
        return self.vehicles[self.current_v_idx].frames[self.current_frame_idx][1]

    def set_interval_start(self, evt):
        t = self.get_current_time()
        if evt == 'turn': self.temp_turn_start = t
        else: self.temp_brake_start = t

    def save_interval(self, evt):
        t_end = self.get_current_time()
        v = self.vehicles[self.current_v_idx]
        
        if evt == 'turn':
            if self.temp_turn_start is None: return
            t_start, event_type = self.temp_turn_start, self.turn_group.checkedButton().text()
            self.temp_turn_start = None
            if event_type == "off": return
        else:
            if self.temp_brake_start is None: return
            t_start, event_type = self.temp_brake_start, "brake_on"
            self.temp_brake_start = None
            
        if t_end < t_start: t_start, t_end = t_end, t_start
        v.video_intervals.append({'type': event_type, 'start': t_start, 'end': t_end})
        v.save_video_labels()
        self.refresh_interval_list()

    def refresh_interval_list(self):
        self.list_intervals.clear()
        v = self.vehicles[self.current_v_idx]
        for idx, i in enumerate(v.video_intervals):
            self.list_intervals.addItem(f"{idx}: {i['type']} ({i['start']:.2f}s - {i['end']:.2f}s)")

    def delete_selected_interval(self):
        row = self.list_intervals.currentRow()
        if row < 0: return
        v = self.vehicles[self.current_v_idx]
        del v.video_intervals[row]
        v.save_video_labels()
        self.refresh_interval_list()

    # --- Frame-wise Methods ---
    def save_current_and_next(self):
        v = self.vehicles[self.current_v_idx]
        f_idx = v.frames[self.current_frame_idx][0]
        v.frame_labels[f_idx] = {
            'turn': self.f_turn_group.checkedButton().text(), 
            'brake': self.f_brake_group.checkedButton().text()
        }
        v.save_frame_labels()
        self.refresh_frame_label_list()
        self.step_frame(1) 

    def copy_prev_frame_label(self):
        v = self.vehicles[self.current_v_idx]
        if not v.frame_labels: return
        f_idx = v.frames[self.current_frame_idx][0]
        prev_keys = [k for k in v.frame_labels.keys() if k < f_idx]
        if not prev_keys: return
        
        last_lbl = v.frame_labels[max(prev_keys)]
        for btn in self.f_turn_group.buttons():
            if btn.text() == last_lbl['turn']: btn.setChecked(True)
        for btn in self.f_brake_group.buttons():
            if btn.text() == last_lbl['brake']: btn.setChecked(True)
        self.save_current_and_next()

    def refresh_frame_label_list(self):
        self.list_frame_labels.clear()
        if not self.vehicles: return
        v = self.vehicles[self.current_v_idx]
        
        for f_idx in sorted(v.frame_labels.keys()):
            lbl = v.frame_labels[f_idx]
            item_text = f"Frame {f_idx:06d}: {lbl['turn']}, {lbl['brake']}"
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, f_idx) 
            self.list_frame_labels.addItem(item)

    def jump_to_frame_from_list(self, item):
        target_f_idx = item.data(QtCore.Qt.UserRole)
        v = self.vehicles[self.current_v_idx]
        for internal_idx, frame_data in enumerate(v.frames):
            if frame_data[0] == target_f_idx:
                self.current_frame_idx = internal_idx
                self.update_display()
                break

    def delete_selected_frame_label(self):
        row = self.list_frame_labels.currentRow()
        if row < 0: return
        item = self.list_frame_labels.item(row)
        target_f_idx = item.data(QtCore.Qt.UserRole)
        
        v = self.vehicles[self.current_v_idx]
        if target_f_idx in v.frame_labels:
            del v.frame_labels[target_f_idx]
            
            img_path = next((path for f, t, path in v.frames if f == target_f_idx), None)
            if img_path:
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    
        self.refresh_frame_label_list()
        self.update_display()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    font = app.font()
    font.setPointSize(12) 
    app.setFont(font)
    
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())