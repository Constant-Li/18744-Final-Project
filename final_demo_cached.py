import os
import sys
import re
import glob
import json
import math
from bisect import bisect_right
import shutil
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models
import pdb

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
DET_MODEL_PATH = r"F:\18744\runs\detect\runs_detect\ft_bestpt_640\weights\best.pt"
TRACKER_CFG = "botsort_reid_vehicle.yaml"
FPS_FALLBACK = 30.0

# ---- Crop saving ----
FRAME_WISE_STRIDE = 1   # Save crops every frame
VIDEO_WISE_STRIDE = 16  # Group size for video-wise subfolders (16 frames per folder)
JPEG_QUALITY = 100      # 0-100

# ---- Thresholds & Tracking ----
DET_CONF = 0.50
# DET_CONF = 0.85
TRUST_CONF = 0.60
IOU_THRES = 0.7
RECOG_INPUT_SIZE = 224

TURN_LABELS = ["off", "left", "right", "both", "unknown"]
BRAKE_LABELS = ["brake_off", "brake_on"]
HEAD_TAIL_LABELS = ["head", "tail"]

# ---- Real-time post-processing (causal / stride-aware) ----
PP_TURN_TAU_SEC = 0.20          # EMA smoothing time constant for turn probabilities
PP_BRAKE_TAU_SEC = 0.12         # EMA smoothing time constant for brake probabilities
PP_TURN_ON_THR = 0.42           # minimum smoothed evidence to activate left/right
PP_TURN_MARGIN = 0.08           # left-right separation margin
PP_BOTH_ON_THR = 0.35           # minimum smoothed evidence to activate both / hazard
PP_TURN_HOLD_SEC = 0.90         # keep turn state alive across short off gaps
PP_BRAKE_ON_THR = 0.60          # brake-on enter threshold
PP_BRAKE_OFF_THR = 0.35         # brake-on exit threshold
PP_BRAKE_HOLD_SEC = 0.25        # keep brake-on alive across brief score dips
PP_LIVE_LOOKBACK_MULTIPLIER = 3.5  # for UI carry-forward on unsampled frames
PP_LIVE_MIN_LOOKBACK_SEC = 0.75


class TLD_resnet(nn.Module):
    def __init__(self, loss_weights=None) -> None:
        super().__init__()
        self.conv2d = getattr(models, "resnet34")(pretrained=True)
        num_ftrs = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Identity()
        self.fc_brake = nn.Linear(num_ftrs, 2)
        self.fc_turn = nn.Linear(num_ftrs, 5)
        self.loss = nn.CrossEntropyLoss()
        self.loss_weights = loss_weights

    def forward(self, inputs_dict):
        x = inputs_dict["x"].permute(0, 3, 1, 2)
        x = self.conv2d(x)
        turn_result = self.fc_turn(x)
        brake_result = self.fc_brake(x)
        return {
            "turn_result": turn_result,
            "brake_result": brake_result,
        }


class TaillightBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TaillightBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        return max_pool

class video_network(nn.Module):
    def __init__(self, loss_weights, transfer_label=False) -> None:
        super().__init__()
        # self.conv2d = getattr(models, 'resnet18')(pretrained=True)
        self.conv2d = getattr(models, 'resnet34')(pretrained=True)
        self.transfer_label = transfer_label
        num_ftrs = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Identity()
        self.temporal_model = TaillightBiLSTM(input_dim=num_ftrs, hidden_dim=num_ftrs//2)
        self.fc_turn = nn.Linear(num_ftrs, 5)
        self.loss_turn = nn.CrossEntropyLoss()
        self.loss_weights = loss_weights
        

    def forward(self, inputs_dict):
        # x = inputs_dict['x'][2]
        # img = x.detach().cpu().numpy()
        # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img_bgr = (img_bgr * 255).clip(0, 255).astype(np.uint8)
        # cv2.imwrite('test.png', img_bgr)
        # pdb.set_trace()
        b, t, h, w, c = inputs_dict['x'].shape
        x = inputs_dict['x'].reshape(b*t, h, w, c)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d(x).reshape(b, t, -1)
        # pdb.set_trace()
        tm_outputs = self.temporal_model(x)
        turn_result = self.fc_turn(tm_outputs)
        # brake_result = self.fc_brake(tm_outputs)
        return{
            'turn_result': turn_result,
            # 'brake_result': brake_result
        }


class HeadTailResnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2d = getattr(models, "resnet34")(pretrained=True)
        num_ftrs = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, 2)

    def forward(self, inputs_dict):
        x = inputs_dict["x"].permute(0, 3, 1, 2)
        x = self.conv2d(x)
        result = self.fc(x)
        return {"result": result}


def pick_torch_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint_state_dict(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


# ==========================================
# TRACKING UTILS & THREAD
# ==========================================
def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def clip_bbox_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def draw_box_id(img, x1, y1, x2, y2, tid, conf):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"id:{tid} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - 8)
    cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text), (0, 255, 0), -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, y_text - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


def get_unique_output_root(base_output_dir, video_name):
    root_parent = os.path.join(base_output_dir, "output")
    os.makedirs(root_parent, exist_ok=True)

    candidate = os.path.join(root_parent, video_name)
    if not os.path.exists(candidate):
        return candidate

    idx = 1
    while True:
        candidate = os.path.join(root_parent, f"{video_name}_{idx:03d}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def normalize_path_for_compare(path):
    if not path:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.path.normpath(path)))
    except Exception:
        return str(path)


def get_video_signature(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    signature = {
        "source_video_path": normalize_path_for_compare(video_path),
        "source_video_name": video_name,
    }
    try:
        stat = os.stat(video_path)
        signature["source_video_size"] = int(stat.st_size)
        signature["source_video_mtime_ns"] = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    except Exception:
        pass
    return signature


def _extract_vis_video_path(output_root, tracking_meta):
    explicit_path = tracking_meta.get("visualization_video_path")
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    vis_dir = os.path.join(output_root, "visualization")
    if not os.path.isdir(vis_dir):
        return None

    mp4s = sorted(glob.glob(os.path.join(vis_dir, "*_track.mp4")))
    if mp4s:
        return mp4s[0]
    return None


def read_existing_tracking_bundle(output_root, video_path):
    meta_path = os.path.join(output_root, "tracking_meta.json")
    frame_wise_root = os.path.join(output_root, "frame-wise")
    if not os.path.isfile(meta_path) or not os.path.isdir(frame_wise_root):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            tracking_meta = json.load(f)
    except Exception:
        return None

    current_sig = get_video_signature(video_path)
    stored_path = normalize_path_for_compare(tracking_meta.get("source_video_path"))
    stored_size = tracking_meta.get("source_video_size")
    stored_mtime = tracking_meta.get("source_video_mtime_ns")
    stored_name = tracking_meta.get("video_name") or tracking_meta.get("source_video_name")

    matched = False
    if stored_path:
        matched = stored_path == current_sig["source_video_path"]
        if matched and stored_size is not None and current_sig.get("source_video_size") is not None:
            matched = int(stored_size) == int(current_sig["source_video_size"])
        if matched and stored_mtime is not None and current_sig.get("source_video_mtime_ns") is not None:
            matched = int(stored_mtime) == int(current_sig["source_video_mtime_ns"])
    else:
        matched = stored_name == current_sig["source_video_name"]

    if not matched:
        return None

    vis_video_path = _extract_vis_video_path(output_root, tracking_meta)
    if not vis_video_path:
        return None

    fps = tracking_meta.get("fps")
    if fps is None:
        cap = cv2.VideoCapture(vis_video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        finally:
            cap.release()

    return {
        "output_root": output_root,
        "vis_video_path": vis_video_path,
        "fps": float(fps) if fps else FPS_FALLBACK,
    }


def find_existing_tracking_bundle(base_output_dir, video_path):
    root_parent = os.path.join(base_output_dir, "output")
    if not os.path.isdir(root_parent):
        return None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    candidates = []
    for output_root in glob.glob(os.path.join(root_parent, f"{video_name}*")):
        if not os.path.isdir(output_root):
            continue
        folder_name = os.path.basename(output_root)
        if folder_name != video_name and not re.fullmatch(rf"{re.escape(video_name)}_\d{{3}}", folder_name):
            continue
        bundle = read_existing_tracking_bundle(output_root, video_path)
        if bundle is not None:
            try:
                bundle["mtime"] = os.path.getmtime(output_root)
            except Exception:
                bundle["mtime"] = 0.0
            candidates.append(bundle)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.get("mtime", 0.0), reverse=True)
    return candidates[0]


class RealTimeTrackPostProcessor:
    """Causal, stride-aware post-processing for one vehicle track.

    This decoder only uses current and past predictions, so it can run online.
    It does not assume predictions arrive at a fixed frame stride; instead it uses
    the actual frame-number gap to compute elapsed time for smoothing / hysteresis.
    """

    def __init__(
        self,
        fps,
        turn_tau_sec=PP_TURN_TAU_SEC,
        brake_tau_sec=PP_BRAKE_TAU_SEC,
        turn_on_thr=PP_TURN_ON_THR,
        turn_margin=PP_TURN_MARGIN,
        both_on_thr=PP_BOTH_ON_THR,
        turn_hold_sec=PP_TURN_HOLD_SEC,
        brake_on_thr=PP_BRAKE_ON_THR,
        brake_off_thr=PP_BRAKE_OFF_THR,
        brake_hold_sec=PP_BRAKE_HOLD_SEC,
    ):
        self.fps = max(float(fps), 1e-6)
        self.turn_tau_sec = float(turn_tau_sec)
        self.brake_tau_sec = float(brake_tau_sec)
        self.turn_on_thr = float(turn_on_thr)
        self.turn_margin = float(turn_margin)
        self.both_on_thr = float(both_on_thr)
        self.turn_hold_sec = float(turn_hold_sec)
        self.brake_on_thr = float(brake_on_thr)
        self.brake_off_thr = float(brake_off_thr)
        self.brake_hold_sec = float(brake_hold_sec)

        self.prev_frame_no = None
        self.turn_state = "off"
        self.brake_state = "brake_off"
        self.turn_ema = np.zeros(len(TURN_LABELS), dtype=np.float32)
        self.brake_ema = np.zeros(len(BRAKE_LABELS), dtype=np.float32)
        self.last_turn_active_frame = None
        self.last_brake_on_frame = None

    def _alpha(self, dt_sec, tau_sec):
        dt_sec = max(float(dt_sec), 1.0 / self.fps)
        tau_sec = max(float(tau_sec), 1e-6)
        return 1.0 - math.exp(-dt_sec / tau_sec)

    def _turn_candidate(self):
        off_score = float(self.turn_ema[0]) + 0.35 * float(self.turn_ema[4])
        left_score = float(self.turn_ema[1]) + 0.50 * float(self.turn_ema[3])
        right_score = float(self.turn_ema[2]) + 0.50 * float(self.turn_ema[3])
        both_score = float(self.turn_ema[3])

        if both_score >= self.both_on_thr and both_score >= max(left_score, right_score) - self.turn_margin:
            return "both", both_score, {"off": off_score, "left": left_score, "right": right_score, "both": both_score}

        if left_score >= self.turn_on_thr and left_score > off_score and (left_score - right_score) >= self.turn_margin:
            return "left", left_score, {"off": off_score, "left": left_score, "right": right_score, "both": both_score}

        if right_score >= self.turn_on_thr and right_score > off_score and (right_score - left_score) >= self.turn_margin:
            return "right", right_score, {"off": off_score, "left": left_score, "right": right_score, "both": both_score}

        return "off", off_score, {"off": off_score, "left": left_score, "right": right_score, "both": both_score}

    def update(self, frame_no, turn_scores, brake_scores):
        frame_no = int(frame_no)
        turn_scores = np.asarray(turn_scores, dtype=np.float32)
        brake_scores = np.asarray(brake_scores, dtype=np.float32)

        if self.prev_frame_no is None:
            dt_sec = 1.0 / self.fps
            self.turn_ema = turn_scores.copy()
            self.brake_ema = brake_scores.copy()
        else:
            gap_frames = max(1, frame_no - self.prev_frame_no)
            dt_sec = gap_frames / self.fps
            a_turn = self._alpha(dt_sec, self.turn_tau_sec)
            a_brake = self._alpha(dt_sec, self.brake_tau_sec)
            self.turn_ema = (1.0 - a_turn) * self.turn_ema + a_turn * turn_scores
            self.brake_ema = (1.0 - a_brake) * self.brake_ema + a_brake * brake_scores

        turn_candidate, turn_conf, turn_evidence = self._turn_candidate()
        if turn_candidate != "off":
            self.turn_state = turn_candidate
            self.last_turn_active_frame = frame_no
        else:
            if self.turn_state != "off" and self.last_turn_active_frame is not None:
                turn_gap_sec = (frame_no - self.last_turn_active_frame) / self.fps
                if turn_gap_sec > self.turn_hold_sec:
                    self.turn_state = "off"
            else:
                self.turn_state = "off"

        brake_on_prob = float(self.brake_ema[1])
        if self.brake_state == "brake_off":
            if brake_on_prob >= self.brake_on_thr:
                self.brake_state = "brake_on"
                self.last_brake_on_frame = frame_no
        else:
            if brake_on_prob >= self.brake_off_thr:
                self.last_brake_on_frame = frame_no
            elif self.last_brake_on_frame is None:
                self.brake_state = "brake_off"
            else:
                brake_gap_sec = (frame_no - self.last_brake_on_frame) / self.fps
                if brake_gap_sec > self.brake_hold_sec:
                    self.brake_state = "brake_off"

        self.prev_frame_no = frame_no
        return {
            "turn_label": self.turn_state,
            "brake_label": self.brake_state,
            "turn_scores": [float(v) for v in self.turn_ema.tolist()],
            "brake_scores": [float(v) for v in self.brake_ema.tolist()],
            "turn_evidence": {k: float(v) for k, v in turn_evidence.items()},
            "turn_conf": float(turn_conf),
            "brake_conf": float(brake_on_prob if self.brake_state == "brake_on" else 1.0 - brake_on_prob),
        }


def build_postprocess_predictions(frame_predictions, fps):
    grouped = {}
    for pred in frame_predictions.values():
        track_id = safe_int(pred.get("track_id"), default=None)
        frame_no = safe_int(pred.get("frame_no"), default=None)
        if track_id is None or frame_no is None:
            continue
        grouped.setdefault(track_id, []).append(pred)

    pp_predictions = {}
    for track_id, items in grouped.items():
        items.sort(key=lambda x: x.get("frame_no", -1))
        pp = RealTimeTrackPostProcessor(fps=fps)
        for item in items:
            frame_no = int(item["frame_no"])
            pp_out = pp.update(frame_no, item["turn_scores"], item["brake_scores"])
            pp_predictions[f"{track_id}_{frame_no}"] = {
                "track_id": track_id,
                "frame_no": frame_no,
                "turn_label": pp_out["turn_label"],
                "brake_label": pp_out["brake_label"],
                "turn_scores": pp_out["turn_scores"],
                "brake_scores": pp_out["brake_scores"],
                "turn_evidence": pp_out["turn_evidence"],
                "turn_conf": pp_out["turn_conf"],
                "brake_conf": pp_out["brake_conf"],
                "source": "post_process",
            }
    return pp_predictions


def build_prediction_index(predictions):
    grouped = {}
    for pred in predictions.values():
        track_id = safe_int(pred.get("track_id"), default=None)
        frame_no = safe_int(pred.get("frame_no"), default=None)
        if track_id is None or frame_no is None:
            continue
        grouped.setdefault(track_id, []).append((frame_no, pred))

    index = {}
    for track_id, items in grouped.items():
        items.sort(key=lambda x: x[0])
        frame_nos = [frame_no for frame_no, _ in items]
        preds = [pred for _, pred in items]
        gaps = [b - a for a, b in zip(frame_nos[:-1], frame_nos[1:]) if b > a]
        median_gap = int(np.median(gaps)) if gaps else 1
        index[track_id] = {
            "frames": frame_nos,
            "preds": preds,
            "median_gap": max(1, median_gap),
        }
    return index


def get_latest_prediction_at_or_before(index, track_id, frame_no, fps, carry_forward=False):
    if not index:
        return None
    track_id = safe_int(track_id, default=None)
    if track_id is None or track_id not in index:
        return None
    entry = index[track_id]
    frames = entry["frames"]
    pos = bisect_right(frames, int(frame_no)) - 1
    if pos < 0:
        return None
    pred_frame = frames[pos]
    pred = entry["preds"][pos]
    if not carry_forward:
        return pred if pred_frame == int(frame_no) else None

    max_age_frames = max(
        int(entry["median_gap"] * PP_LIVE_LOOKBACK_MULTIPLIER),
        int(max(float(fps), 1.0) * PP_LIVE_MIN_LOOKBACK_SEC),
    )
    if int(frame_no) - pred_frame <= max_age_frames:
        return pred
    return None


class TrackerThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, int)
    finished_tracking = QtCore.pyqtSignal(str, str, float)
    tracking_error = QtCore.pyqtSignal(str)

    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        existing_bundle = find_existing_tracking_bundle(self.output_dir, self.video_path)
        if existing_bundle is not None:
            self.progress_update.emit(1, 1)
            self.finished_tracking.emit(
                existing_bundle["output_root"],
                existing_bundle["vis_video_path"],
                existing_bundle["fps"],
            )
            return

        output_root = get_unique_output_root(self.output_dir, video_name)
        frame_wise_root = os.path.join(output_root, "frame-wise")
        vis_root = os.path.join(output_root, "visualization")
        os.makedirs(frame_wise_root, exist_ok=True)
        os.makedirs(vis_root, exist_ok=True)

        try:
            model = YOLO(DET_MODEL_PATH)
        except Exception as e:
            self.tracking_error.emit(f"Error loading model: {e}")
            return

        model_task = getattr(model, "task", None)
        if model_task is not None and model_task not in ("detect", "segment", "pose", "obb"):
            self.tracking_error.emit(
                f"Wrong model task '{model_task}'. Please use a detection model for tracking."
            )
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.tracking_error.emit(f"Cannot open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vis_video_path = os.path.join(vis_root, f"{video_name}_track.mp4")
        writer = cv2.VideoWriter(vis_video_path, fourcc, fps, (w, h))

        frame_idx = -1
        tracking_meta = {
            "video_name": video_name,
            "fps": float(fps),
            "visualization_video_path": vis_video_path,
            **get_video_signature(self.video_path),
            "frames": {},
        }
        while self.is_running:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                continue
            frame_idx += 1

            if frame_idx % 10 == 0:
                self.progress_update.emit(frame_idx, total_frames)

            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER_CFG,
                    conf=DET_CONF,
                    iou=IOU_THRES,
                    device="cuda",
                    verbose=False,
                )[0]
            except Exception as e:
                print(f"[WARN] track failed at frame {frame_idx}: {e}")
                continue

            annotated = frame.copy()
            frame_tracks = []
            if results.boxes is not None and results.boxes.xyxy is not None:
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None
                ids = results.boxes.id.cpu().numpy() if getattr(results.boxes, "id", None) is not None else None

                for i in range(len(xyxy)):
                    tid = safe_int(ids[i]) if ids is not None else None
                    if tid is None:
                        continue

                    conf = float(confs[i]) if confs is not None else 1.0
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    x1, y1, x2, y2 = clip_bbox_xyxy(x1, y1, x2, y2, w, h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    draw_box_id(annotated, x1, y1, x2, y2, tid, conf)
                    frame_tracks.append(
                        {
                            "track_id": tid,
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                        }
                    )

                    if conf < TRUST_CONF:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    frame_wise_id_dir = os.path.join(frame_wise_root, str(tid))
                    os.makedirs(frame_wise_id_dir, exist_ok=True)
                    frame_wise_path = os.path.join(
                        frame_wise_id_dir,
                        f"{video_name}_id{tid}_frame{frame_idx:06d}.jpg",
                    )
                    cv2.imwrite(frame_wise_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            writer.write(annotated)
            tracking_meta["frames"][str(frame_idx)] = frame_tracks

        cap.release()
        writer.release()
        meta_path = os.path.join(output_root, "tracking_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(tracking_meta, f)
        self.progress_update.emit(total_frames, total_frames)
        self.finished_tracking.emit(output_root, vis_video_path, fps)

    def stop(self):
        self.is_running = False


class RecognitionThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, int)
    finished_recognition = QtCore.pyqtSignal(dict, str)

    def __init__(self, frame_wise_root, weights_path):
        super().__init__()
        self.frame_wise_root = frame_wise_root
        self.weights_path = weights_path
        self.is_running = True

    def _load_model(self, device):
        model = TLD_resnet(loss_weights=None)
        state_dict = load_checkpoint_state_dict(self.weights_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

    def _preprocess(self, img_bgr, device):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (RECOG_INPUT_SIZE, RECOG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img_rgb).float() / 127.5 - 1
        # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        # std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        # x = (x - mean) / std
        x = x.unsqueeze(0).to(device)
        return x

    def run(self):
        if not self.frame_wise_root or not os.path.isdir(self.frame_wise_root):
            self.finished_recognition.emit({}, "frame-wise folder not found")
            return
        if not self.weights_path or not os.path.isfile(self.weights_path):
            self.finished_recognition.emit({}, "model weights not found")
            return

        image_files = sorted(glob.glob(os.path.join(self.frame_wise_root, "*", "*.jpg")))
        total = len(image_files)
        if total == 0:
            self.finished_recognition.emit({}, "no frame-wise crop images found")
            return

        device = pick_torch_device()
        try:
            model = self._load_model(device)
        except Exception as e:
            self.finished_recognition.emit({}, f"failed to load model: {e}")
            return

        predictions = {}
        infer_fail = 0
        skipped_parse = 0
        with torch.no_grad():
            for idx, img_path in enumerate(image_files):
                if not self.is_running:
                    break

                img = cv2.imread(img_path)
                if img is None:
                    continue
                # pdb.set_trace()

                try:
                    # pdb.set_trace()
                    x = self._preprocess(img, device)
                    # pdb.set_trace()
                    out = model({"x": x})
                    turn_logits = out["turn_result"][0]
                    brake_logits = out["brake_result"][0]
                    turn_idx = int(torch.argmax(turn_logits).item())
                    brake_idx = int(torch.argmax(brake_logits).item())
                    turn_scores = torch.softmax(turn_logits, dim=0).detach().cpu().tolist()
                    brake_scores = torch.softmax(brake_logits, dim=0).detach().cpu().tolist()

                    folder = os.path.basename(os.path.dirname(img_path))
                    if folder.isdigit():
                        track_id = int(folder)
                    else:
                        m_tid = re.search(r"(\d+)", folder)
                        track_id = int(m_tid.group(1)) if m_tid else -1
                    m = re.search(r"frame(\d+)", os.path.basename(img_path))
                    frame_no = int(m.group(1)) if m else -1
                    if track_id < 0 or frame_no < 0:
                        skipped_parse += 1
                        continue

                    predictions[f"{track_id}_{frame_no}"] = {
                        "track_id": track_id,
                        "frame_no": frame_no,
                        "turn_idx": turn_idx,
                        "brake_idx": brake_idx,
                        "turn_label": TURN_LABELS[turn_idx],
                        "brake_label": BRAKE_LABELS[brake_idx],
                        "turn_scores": turn_scores,
                        "brake_scores": brake_scores,
                    }
                except Exception:
                    infer_fail += 1
                    continue

                if idx % 20 == 0:
                    self.progress_update.emit(idx + 1, total)

        self.progress_update.emit(total, total)
        status_msg = f"ok | total={total} pred={len(predictions)} infer_fail={infer_fail} parse_skip={skipped_parse}"
        self.finished_recognition.emit(predictions, status_msg)

    def stop(self):
        self.is_running = False


class VideoWiseRecognitionThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, int)
    finished_recognition = QtCore.pyqtSignal(dict, str)

    def __init__(self, frame_wise_root, weights_path):
        super().__init__()
        self.frame_wise_root = frame_wise_root
        self.weights_path = weights_path
        self.is_running = True

    def _load_model(self, device):
        model = video_network(loss_weights=None, transfer_label=True)
        state_dict = load_checkpoint_state_dict(self.weights_path, device)
        model.load_state_dict(state_dict, strict=True)
        # ckpt = torch.load(self.weights_path, map_location=device)
        # if isinstance(ckpt, dict):
        #     if "model_state_dict" in ckpt:
        #         state_dict = ckpt["model_state_dict"]
        #     elif "state_dict" in ckpt:
        #         state_dict = ckpt["state_dict"]
        #     else:
        #         state_dict = ckpt
        # else:
        #     state_dict = ckpt
        model.to(device)
        model.eval()
        return model

    def _preprocess_frame(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (RECOG_INPUT_SIZE, RECOG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
        return x

    def run(self):
        if not self.frame_wise_root or not os.path.isdir(self.frame_wise_root):
            self.finished_recognition.emit({}, "frame-wise folder not found")
            return
        track_dirs = sorted(glob.glob(os.path.join(self.frame_wise_root, "*")))
        if not self.weights_path or not os.path.isfile(self.weights_path):
            self.finished_recognition.emit({}, "model weights not found")
            return

        # chunk_dirs = sorted(glob.glob(os.path.join(self.frame_wise_root, "*", "chunk_*")))
        # total = len(chunk_dirs)
        # if total == 0:
        #     self.finished_recognition.emit({}, "no video-wise chunks found")
        #     return

        device = pick_torch_device()
        try:
            model = self._load_model(device)
        except Exception as e:
            self.finished_recognition.emit({}, f"failed to load video model: {e}")
            return
        
        predictions = {}
        infer_fail = 0
        parse_skip = 0
        idx = 0
        total = 0

        track_frame_counts = []

        for track_dir in track_dirs:
            image_paths = glob.glob(os.path.join(track_dir, "*.jpg"))
            num_frames = len(image_paths)
            track_frame_counts.append(num_frames)

        total = sum(max(0, n - 14) for n in track_frame_counts)

        with torch.no_grad():
            for track_dir in track_dirs:
                if not self.is_running:
                    break
                # num_frames = len(glob.glob(os.path.join(track_dir, "*.jpg")))
                # total += max(0, num_frames - 14)

                track_name = os.path.basename(track_dir)
                if track_name.isdigit():
                    track_id = int(track_name)
                else:
                    continue

                image_paths = sorted(glob.glob(os.path.join(track_dir, "*.jpg")))

                frames = []
                for img_path in image_paths:
                    m = re.search(r"frame(\d+)", os.path.basename(img_path))
                    if not m:
                        continue
                    frame_no = int(m.group(1))
                    frames.append((frame_no, img_path))

                frames.sort(key=lambda x: x[0])

                for i in range(len(frames)):
                    if i < 7 or i >= len(frames) - 7:
                        continue

                    seq = frames[i-7:i+8]

                    seq_tensors = []
                    valid = True

                    for _, img_path in seq:
                        img = cv2.imread(img_path)
                        if img is None:
                            valid = False
                            break
                        seq_tensors.append(self._preprocess_frame(img))

                    if not valid:
                        continue

                    try:
                        x = torch.stack(seq_tensors, dim=0).unsqueeze(0).to(device)

                        out = model({"x": x})
                        turn_logits = out["turn_result"][0]

                        turn_idx = int(torch.argmax(turn_logits).item())
                        turn_scores = torch.softmax(turn_logits, dim=0).cpu().tolist()

                        center_frame_no = frames[i][0]

                        predictions[f"{track_id}_{center_frame_no}"] = {
                            "track_id": track_id,
                            "frame_no": center_frame_no,
                            "turn_idx": turn_idx,
                            "turn_label": TURN_LABELS[turn_idx],
                            "turn_scores": turn_scores,
                            "source": "video_wise_window",
                        }

                    except Exception:
                        infer_fail += 1
                        continue
                    idx += 1
                    if idx % 10 == 0:
                        self.progress_update.emit(idx+1, total)

        self.progress_update.emit(total, total)
        status = (
            f"ok | windows={total} pred={len(predictions)} "
            f"infer_fail={infer_fail} parse_skip={parse_skip}"
        )
        self.finished_recognition.emit(predictions, status)

    def stop(self):
        self.is_running = False


class HeadTailRecognitionThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, int)
    finished_recognition = QtCore.pyqtSignal(dict, str)

    def __init__(self, frame_wise_root, weights_path):
        super().__init__()
        self.frame_wise_root = frame_wise_root
        self.weights_path = weights_path
        self.is_running = True

    def _load_model(self, device):
        model = HeadTailResnet()
        state_dict = load_checkpoint_state_dict(self.weights_path, device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model

    def _preprocess(self, img_bgr, device):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (RECOG_INPUT_SIZE, RECOG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img_rgb).float() / 127.5 - 1
        return x.unsqueeze(0).to(device)

    def run(self):
        if not self.frame_wise_root or not os.path.isdir(self.frame_wise_root):
            self.finished_recognition.emit({}, "frame-wise folder not found")
            return
        if not self.weights_path or not os.path.isfile(self.weights_path):
            self.finished_recognition.emit({}, "model weights not found")
            return

        image_files = sorted(glob.glob(os.path.join(self.frame_wise_root, "*", "*.jpg")))
        total = len(image_files)
        if total == 0:
            self.finished_recognition.emit({}, "no frame-wise crop images found")
            return

        device = pick_torch_device()
        try:
            model = self._load_model(device)
        except Exception as e:
            self.finished_recognition.emit({}, f"failed to load head-tail model: {e}")
            return

        predictions = {}
        infer_fail = 0
        skipped_parse = 0
        with torch.no_grad():
            for idx, img_path in enumerate(image_files):
                if not self.is_running:
                    break

                img = cv2.imread(img_path)
                if img is None:
                    continue

                try:
                    x = self._preprocess(img, device)
                    out = model({"x": x})
                    logits = out.get("result")
                    if logits is None:
                        logits = out.get("head_tail_result")
                    if logits is None:
                        raise KeyError("missing result tensor")

                    logits = logits[0]
                    pred_idx = int(torch.argmax(logits).item())
                    scores = torch.softmax(logits, dim=0).detach().cpu().tolist()

                    folder = os.path.basename(os.path.dirname(img_path))
                    if folder.isdigit():
                        track_id = int(folder)
                    else:
                        m_tid = re.search(r"(\d+)", folder)
                        track_id = int(m_tid.group(1)) if m_tid else -1
                    m = re.search(r"frame(\d+)", os.path.basename(img_path))
                    frame_no = int(m.group(1)) if m else -1
                    if track_id < 0 or frame_no < 0:
                        skipped_parse += 1
                        continue

                    predictions[f"{track_id}_{frame_no}"] = {
                        "track_id": track_id,
                        "frame_no": frame_no,
                        "head_tail_idx": pred_idx,
                        "head_tail_label": HEAD_TAIL_LABELS[pred_idx],
                        "head_tail_scores": scores,
                    }
                except Exception:
                    infer_fail += 1
                    continue

                if idx % 20 == 0:
                    self.progress_update.emit(idx + 1, total)

        self.progress_update.emit(total, total)
        status_msg = (
            f"ok | total={total} pred={len(predictions)} "
            f"infer_fail={infer_fail} parse_skip={skipped_parse}"
        )
        self.finished_recognition.emit(predictions, status_msg)

    def stop(self):
        self.is_running = False


# ==========================================
# VISUALIZATION APP (NO LABEL UI)
# ==========================================
class LabelingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detect + Track Visualizer")
        self.resize(1100, 760)

        self.fps = FPS_FALLBACK
        self.tracker_thread = None

        self.video_cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.is_playing = False
        self.frame_wise_root = None
        self.video_wise_root = None

        self.vehicles = []
        self.current_vehicle_idx = -1
        self.current_vehicle_frame_idx = 0
        self.is_crop_playing = False
        self.recog_thread = None
        self.video_recog_thread = None
        self.head_tail_recog_thread = None
        self.recog_weights_path = ""
        self.video_recog_weights_path = ""
        self.head_tail_weights_path = ""
        self.recognition_results = {}
        self.video_recognition_results = {}
        self.head_tail_recognition_results = {}
        self.postprocess_results = {}
        self.pp_index = {}
        self.tracking_meta = {}

        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        top_bar = QtWidgets.QHBoxLayout()
        btn_load_video = QtWidgets.QPushButton("Select Video and Run Detect+Track")
        btn_load_video.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        btn_load_video.clicked.connect(self.select_and_track_video)
        btn_recognition = QtWidgets.QPushButton("Frame-wise Recognition")
        btn_recognition.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_recognition.clicked.connect(self.run_frame_wise_recognition)
        btn_video_recognition = QtWidgets.QPushButton("Video-wise Recognition")
        btn_video_recognition.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        btn_video_recognition.clicked.connect(self.run_video_wise_recognition)
        btn_head_tail_recognition = QtWidgets.QPushButton("Head/Tail Recognition")
        btn_head_tail_recognition.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;")
        btn_head_tail_recognition.clicked.connect(self.run_head_tail_recognition)

        self.lbl_status = QtWidgets.QLabel("Status: Waiting for video...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        top_bar.addWidget(btn_load_video)
        top_bar.addWidget(btn_recognition)
        top_bar.addWidget(btn_video_recognition)
        top_bar.addWidget(btn_head_tail_recognition)
        top_bar.addWidget(self.lbl_status)
        top_bar.addWidget(self.progress_bar, stretch=1)
        main_layout.addLayout(top_bar)

        # Left panel: tracked video visualization
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QtWidgets.QLabel("Run tracking to preview visualization video.")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.image_label, stretch=1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.on_slider_move)
        left_layout.addWidget(self.slider)

        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play / Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_step_b = QtWidgets.QPushButton("< Step Back")
        btn_step_b.clicked.connect(lambda: self.step_frame(-1))
        btn_step_f = QtWidgets.QPushButton("Step Forward >")
        btn_step_f.clicked.connect(lambda: self.step_frame(1))

        self.lbl_time = QtWidgets.QLabel("Time: 0.00s | Frame: 0/0")
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(btn_step_b)
        ctrl_layout.addWidget(btn_step_f)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_time)
        left_layout.addLayout(ctrl_layout)

        # Right panel: per-vehicle crops
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_vehicle = QtWidgets.QLabel("Vehicle: 0 / 0")
        right_layout.addWidget(self.lbl_vehicle)

        nav_layout = QtWidgets.QHBoxLayout()
        btn_prev_vehicle = QtWidgets.QPushButton("<< Prev Vehicle")
        btn_prev_vehicle.clicked.connect(lambda: self.switch_vehicle(-1))
        btn_next_vehicle = QtWidgets.QPushButton("Next Vehicle >>")
        btn_next_vehicle.clicked.connect(lambda: self.switch_vehicle(1))
        nav_layout.addWidget(btn_prev_vehicle)
        nav_layout.addWidget(btn_next_vehicle)
        right_layout.addLayout(nav_layout)

        self.crop_label = QtWidgets.QLabel("No crop data yet.")
        self.crop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.crop_label.setMinimumSize(280, 280)
        self.crop_label.setStyleSheet("background-color: #111; color: white;")

        self.crop_pred_label = QtWidgets.QLabel("Recognition:\nFW turn: -\nFW brake: -\nVW turn: -\nPP turn: -\nPP brake: -\nHT: -")
        self.crop_pred_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.crop_pred_label.setMinimumWidth(160)
        self.crop_pred_label.setStyleSheet("background-color: #1f1f1f; color: #e8e8e8; padding: 8px;")

        crop_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        crop_splitter.addWidget(self.crop_label)
        crop_splitter.addWidget(self.crop_pred_label)
        crop_splitter.setStretchFactor(0, 3)
        crop_splitter.setStretchFactor(1, 1)
        crop_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(crop_splitter, stretch=1)

        self.crop_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.crop_slider.setRange(0, 0)
        self.crop_slider.sliderMoved.connect(self.on_crop_slider_move)
        right_layout.addWidget(self.crop_slider)

        crop_ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_crop_play = QtWidgets.QPushButton("Crop Play / Pause")
        self.btn_crop_play.clicked.connect(self.toggle_crop_play)
        btn_crop_step_b = QtWidgets.QPushButton("< Crop Step")
        btn_crop_step_b.clicked.connect(lambda: self.step_crop_frame(-1))
        btn_crop_step_f = QtWidgets.QPushButton("Crop Step >")
        btn_crop_step_f.clicked.connect(lambda: self.step_crop_frame(1))
        crop_ctrl_layout.addWidget(self.btn_crop_play)
        crop_ctrl_layout.addWidget(btn_crop_step_b)
        crop_ctrl_layout.addWidget(btn_crop_step_f)
        right_layout.addLayout(crop_ctrl_layout)

        self.lbl_crop_time = QtWidgets.QLabel("Crop Frame: 0/0")
        right_layout.addWidget(self.lbl_crop_time)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setChildrenCollapsible(False)

        main_layout.addWidget(main_splitter, stretch=1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.step_frame(1, loop=False))
        self.crop_timer = QtCore.QTimer()
        self.crop_timer.timeout.connect(lambda: self.step_crop_frame(1, loop=False))

    def select_and_track_video(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if not video_path:
            return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if not output_dir:
            return

        self.stop_playback()
        self.lbl_status.setText(f"Tracking: {os.path.basename(video_path)}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.tracker_thread = TrackerThread(video_path, output_dir)
        self.tracker_thread.progress_update.connect(self.update_progress)
        self.tracker_thread.finished_tracking.connect(self.on_tracking_finished)
        self.tracker_thread.tracking_error.connect(self.on_tracking_error)
        self.tracker_thread.start()

    def update_progress(self, current, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_tracking_finished(self, output_root, vis_video_path, fps):
        self.fps = fps if fps > 0 else FPS_FALLBACK
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Done. Visualization: {vis_video_path} | Output: {output_root}")
        self.frame_wise_root = os.path.join(output_root, "frame-wise")
        self.video_wise_root = self.frame_wise_root
        self.recognition_results = {}
        self.video_recognition_results = {}
        self.head_tail_recognition_results = {}
        self.postprocess_results = {}
        self.pp_index = {}
        self.load_tracking_meta(output_root)
        self.load_saved_recognition_outputs(output_root)
        self.load_visualization_video(vis_video_path)
        self.load_vehicle_crops()

    def on_tracking_error(self, message):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Tracking failed: {message}")

    def load_tracking_meta(self, output_root):
        self.tracking_meta = {}
        meta_path = os.path.join(output_root, "tracking_meta.json")
        if not os.path.isfile(meta_path):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self.tracking_meta = json.load(f)
        except Exception:
            self.tracking_meta = {}

    def _load_json_if_exists(self, path):
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def load_saved_recognition_outputs(self, output_root):
        self.recognition_results = self._load_json_if_exists(
            os.path.join(output_root, "frame_wise_recognition.json")
        )
        self.video_recognition_results = self._load_json_if_exists(
            os.path.join(output_root, "video_wise_recognition.json")
        )
        self.head_tail_recognition_results = self._load_json_if_exists(
            os.path.join(output_root, "head_tail_recognition.json")
        )
        self.postprocess_results = self._load_json_if_exists(
            os.path.join(output_root, "postprocess_recognition.json")
        )
        self.pp_index = build_prediction_index(self.postprocess_results)

    def run_frame_wise_recognition(self):
        if not self.frame_wise_root or not os.path.isdir(self.frame_wise_root):
            self.lbl_status.setText("Recognition failed: frame-wise folder not ready.")
            return

        weights_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select recognition model weights",
            self.recog_weights_path or "",
            "Model files (*.pt *.pth *.ckpt);;All files (*)",
        )
        if not weights_path:
            return
        self.recog_weights_path = weights_path

        if self.recog_thread is not None and self.recog_thread.isRunning():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Running frame-wise recognition...")

        self.recog_thread = RecognitionThread(self.frame_wise_root, self.recog_weights_path)
        self.recog_thread.progress_update.connect(self.update_progress)
        self.recog_thread.finished_recognition.connect(self.on_recognition_finished)
        self.recog_thread.start()

    def on_recognition_finished(self, predictions, status):
        self.progress_bar.setVisible(False)
        if not status.startswith("ok"):
            self.lbl_status.setText(f"Recognition failed: {status}")
            return

        self.recognition_results = predictions
        self.postprocess_results = build_postprocess_predictions(self.recognition_results, self.fps)
        self.pp_index = build_prediction_index(self.postprocess_results)
        if self.frame_wise_root:
            out_root = os.path.dirname(self.frame_wise_root)
            out_path = os.path.join(out_root, "frame_wise_recognition.json")
            pp_path = os.path.join(out_root, "postprocess_recognition.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(self.recognition_results, f)
                with open(pp_path, "w", encoding="utf-8") as f:
                    json.dump(self.postprocess_results, f)
            except Exception:
                pass

        self.lbl_status.setText(
            f"Recognition done: {status} | pp={len(self.postprocess_results)} realtime-causal outputs"
        )
        self.show_frame(self.current_frame_idx)
        self.update_vehicle_crop_display()

    def run_video_wise_recognition(self):
        if not self.video_wise_root or not os.path.isdir(self.video_wise_root):
            self.lbl_status.setText("Recognition failed: video-wise folder not ready.")
            return

        weights_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video-wise recognition model weights",
            self.video_recog_weights_path or "",
            "Model files (*.pt *.pth *.ckpt);;All files (*)",
        )
        if not weights_path:
            return
        self.video_recog_weights_path = weights_path

        if self.video_recog_thread is not None and self.video_recog_thread.isRunning():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Running video-wise recognition...")

        self.video_recog_thread = VideoWiseRecognitionThread(self.video_wise_root, self.video_recog_weights_path)
        self.video_recog_thread.progress_update.connect(self.update_progress)
        self.video_recog_thread.finished_recognition.connect(self.on_video_recognition_finished)
        self.video_recog_thread.start()

    def on_video_recognition_finished(self, predictions, status):
        self.progress_bar.setVisible(False)
        if not status.startswith("ok"):
            self.lbl_status.setText(f"Video-wise recognition failed: {status}")
            return

        self.video_recognition_results = predictions
        if self.video_wise_root:
            out_path = os.path.join(os.path.dirname(self.video_wise_root), "video_wise_recognition.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(self.video_recognition_results, f)
            except Exception:
                pass

        self.lbl_status.setText(f"Video-wise recognition done: {status}")
        self.show_frame(self.current_frame_idx)
        self.update_vehicle_crop_display()

    def run_head_tail_recognition(self):
        if not self.frame_wise_root or not os.path.isdir(self.frame_wise_root):
            self.lbl_status.setText("Recognition failed: frame-wise folder not ready.")
            return

        weights_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select head-tail recognition model weights",
            self.head_tail_weights_path or "",
            "Model files (*.pt *.pth *.ckpt);;All files (*)",
        )
        if not weights_path:
            return
        self.head_tail_weights_path = weights_path

        if self.head_tail_recog_thread is not None and self.head_tail_recog_thread.isRunning():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Running head-tail recognition...")

        self.head_tail_recog_thread = HeadTailRecognitionThread(
            self.frame_wise_root,
            self.head_tail_weights_path,
        )
        self.head_tail_recog_thread.progress_update.connect(self.update_progress)
        self.head_tail_recog_thread.finished_recognition.connect(self.on_head_tail_recognition_finished)
        self.head_tail_recog_thread.start()

    def on_head_tail_recognition_finished(self, predictions, status):
        self.progress_bar.setVisible(False)
        if not status.startswith("ok"):
            self.lbl_status.setText(f"Head-tail recognition failed: {status}")
            return

        self.head_tail_recognition_results = predictions
        if self.frame_wise_root:
            out_path = os.path.join(os.path.dirname(self.frame_wise_root), "head_tail_recognition.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(self.head_tail_recognition_results, f)
            except Exception:
                pass

        self.lbl_status.setText(f"Head-tail recognition done: {status}")
        self.show_frame(self.current_frame_idx)
        self.update_vehicle_crop_display()

    def get_postprocess_prediction_for_display(self, track_id, frame_no):
        return get_latest_prediction_at_or_before(
            self.pp_index, track_id, frame_no, self.fps, carry_forward=True
        )

    def load_vehicle_crops(self):
        self.vehicles = []
        self.current_vehicle_idx = -1
        self.current_vehicle_frame_idx = 0

        if not self.frame_wise_root or not os.path.isdir(self.frame_wise_root):
            self.update_vehicle_status()
            return

        folders = sorted(
            [f for f in glob.glob(os.path.join(self.frame_wise_root, "*")) if os.path.isdir(f)],
            key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else os.path.basename(p),
        )

        for folder in folders:
            track_name = os.path.basename(folder)
            images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            frames = []
            for img in images:
                m = re.search(r"frame(\d+)", os.path.basename(img))
                frame_no = int(m.group(1)) if m else -1
                frames.append((frame_no, img))
            if frames:
                frames.sort(key=lambda x: x[0])
                self.vehicles.append({"track_id": track_name, "frames": frames})

        if self.vehicles:
            self.current_vehicle_idx = 0
            self.current_vehicle_frame_idx = 0
            self.update_vehicle_crop_display()
        else:
            self.update_vehicle_status()

    def load_visualization_video(self, vis_video_path):
        self.release_video_cap()

        self.video_cap = cv2.VideoCapture(vis_video_path)
        if not self.video_cap.isOpened():
            self.lbl_status.setText(f"Failed to open visualization video: {vis_video_path}")
            return

        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.current_frame_idx = 0
        self.show_frame(0)

    def show_frame(self, frame_idx):
        if self.video_cap is None:
            return

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.video_cap.read()
        if not ok or frame is None:
            return

        self.current_frame_idx = frame_idx
        self.draw_recognition_on_full_frame(frame, frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)

        t_sec = self.current_frame_idx / max(self.fps, 1e-6)
        self.lbl_time.setText(
            f"Time: {t_sec:.2f}s | Frame: {self.current_frame_idx + 1}/{max(1, self.total_frames)}"
        )

    def draw_recognition_on_full_frame(self, frame, frame_idx):
        if (
            not self.recognition_results
            and not self.video_recognition_results
            and not self.postprocess_results
            and not self.head_tail_recognition_results
        ) or not self.tracking_meta:
            return
        frame_tracks = self.tracking_meta.get("frames", {}).get(str(frame_idx), [])
        for item in frame_tracks:
            tid = safe_int(item.get("track_id"), default=None)
            bbox = item.get("bbox", [])
            if tid is None or len(bbox) != 4:
                continue
            fw_pred = self.recognition_results.get(f"{tid}_{frame_idx}")
            vw_pred = self.video_recognition_results.get(f"{tid}_{frame_idx}")
            pp_pred = self.get_postprocess_prediction_for_display(tid, frame_idx)
            ht_pred = self.head_tail_recognition_results.get(f"{tid}_{frame_idx}")
            if not fw_pred and not vw_pred and not pp_pred and not ht_pred:
                continue
            x1, y1, _, _ = [int(v) for v in bbox]
            lines = []
            if fw_pred:
                lines.append((f"FW:{fw_pred['turn_label']}|{fw_pred['brake_label']}", (0, 255, 255)))
            if vw_pred:
                lines.append((f"VW:{vw_pred['turn_label']}", (255, 200, 0)))
            if pp_pred:
                lines.append((f"PP:{pp_pred['turn_label']}|{pp_pred['brake_label']}", (120, 255, 120)))
            if ht_pred:
                lines.append((f"HT:{ht_pred['head_tail_label']}", (220, 120, 255)))
            line_h = 18
            max_w = 0
            for line, _ in lines:
                (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                max_w = max(max_w, tw)
            ty = max(0, y1 - (line_h * len(lines) + 8))
            cv2.rectangle(frame, (x1, ty), (x1 + max_w + 10, ty + line_h * len(lines) + 6), (30, 30, 30), -1)
            for i, (line, color) in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (x1 + 4, ty + 16 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

    def on_slider_move(self, val):
        self.show_frame(val)

    def switch_vehicle(self, step):
        if not self.vehicles:
            return
        self.current_vehicle_idx = (self.current_vehicle_idx + step) % len(self.vehicles)
        self.current_vehicle_frame_idx = 0
        self.update_vehicle_crop_display()

    def on_crop_slider_move(self, val):
        self.current_vehicle_frame_idx = val
        self.update_vehicle_crop_display()

    def update_vehicle_status(self):
        self.lbl_vehicle.setText("Vehicle: 0 / 0")
        self.crop_label.setPixmap(QtGui.QPixmap())
        self.crop_label.setText("No crop data yet.")
        self.crop_pred_label.setText("Recognition:\nFW turn: -\nFW brake: -\nVW turn: -\nPP turn: -\nPP brake: -\nHT: -")
        if self.is_crop_playing:
            self.toggle_crop_play()
        self.crop_slider.blockSignals(True)
        self.crop_slider.setRange(0, 0)
        self.crop_slider.setValue(0)
        self.crop_slider.blockSignals(False)
        self.lbl_crop_time.setText("Crop Frame: 0/0")

    def update_vehicle_crop_display(self):
        if not self.vehicles or self.current_vehicle_idx < 0:
            self.update_vehicle_status()
            return

        vehicle = self.vehicles[self.current_vehicle_idx]
        frames = vehicle["frames"]
        if not frames:
            self.update_vehicle_status()
            return

        self.current_vehicle_frame_idx = max(0, min(self.current_vehicle_frame_idx, len(frames) - 1))
        frame_no, img_path = frames[self.current_vehicle_frame_idx]

        self.lbl_vehicle.setText(
            f"Vehicle: {self.current_vehicle_idx + 1} / {len(self.vehicles)} | Track ID: {vehicle['track_id']}"
        )

        pixmap = QtGui.QPixmap(img_path)
        if pixmap.isNull():
            self.crop_label.setPixmap(QtGui.QPixmap())
            self.crop_label.setText(f"Failed to load crop: {img_path}")
        else:
            pixmap = pixmap.scaled(
                self.crop_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.crop_label.setPixmap(pixmap)
            self.crop_label.setText("")

        self.crop_slider.blockSignals(True)
        self.crop_slider.setRange(0, len(frames) - 1)
        self.crop_slider.setValue(self.current_vehicle_frame_idx)
        self.crop_slider.blockSignals(False)
        self.lbl_crop_time.setText(
            f"Crop Frame: {self.current_vehicle_frame_idx + 1}/{len(frames)} | Source Frame: {frame_no}"
        )
        track_id = safe_int(vehicle["track_id"], default=-1)
        fw_pred = self.recognition_results.get(f"{track_id}_{frame_no}") if track_id >= 0 else None
        vw_pred = self.video_recognition_results.get(f"{track_id}_{frame_no}") if track_id >= 0 else None
        pp_pred = self.get_postprocess_prediction_for_display(track_id, frame_no) if track_id >= 0 else None
        ht_pred = self.head_tail_recognition_results.get(f"{track_id}_{frame_no}") if track_id >= 0 else None
        fw_text = "FW turn: -\nFW brake: -"
        vw_text = "VW turn: -"
        pp_text = "PP turn: -\nPP brake: -"
        ht_text = "HT: -"
        fw_prob = ""
        pp_prob = ""
        ht_prob = ""
        if fw_pred:
            fw_text = f"FW turn: {fw_pred['turn_label']}\nFW brake: {fw_pred['brake_label']}"
            fw_prob = (
                f"\nFW turn_prob: [{', '.join([f'{v:.2f}' for v in fw_pred['turn_scores']])}]"
                f"\nFW brake_prob: [{', '.join([f'{v:.2f}' for v in fw_pred['brake_scores']])}]"
            )
        if vw_pred:
            vw_text = f"VW turn: {vw_pred['turn_label']}"
        if pp_pred:
            pp_text = f"PP turn: {pp_pred['turn_label']}\nPP brake: {pp_pred['brake_label']}"
            pp_prob = (
                f"\nPP turn_prob: [{', '.join([f'{v:.2f}' for v in pp_pred['turn_scores']])}]"
                f"\nPP brake_prob: [{', '.join([f'{v:.2f}' for v in pp_pred['brake_scores']])}]"
            )
        if ht_pred:
            ht_text = f"HT: {ht_pred['head_tail_label']}"
            ht_prob = f"\nHT prob: [{', '.join([f'{v:.2f}' for v in ht_pred['head_tail_scores']])}]"
        self.crop_pred_label.setText(
            f"Recognition:\n{fw_text}\n{vw_text}\n{pp_text}\n{ht_text}{fw_prob}{pp_prob}{ht_prob}"
        )

    def toggle_crop_play(self):
        if not self.vehicles or self.current_vehicle_idx < 0:
            return

        self.is_crop_playing = not self.is_crop_playing
        if self.is_crop_playing:
            self.btn_crop_play.setText("Crop Pause")
            interval_ms = max(1, int(1000 / max(self.fps, 1.0)))
            self.crop_timer.start(interval_ms)
        else:
            self.btn_crop_play.setText("Crop Play / Pause")
            self.crop_timer.stop()

    def step_crop_frame(self, direction, loop=False):
        if not self.vehicles or self.current_vehicle_idx < 0:
            return

        vehicle = self.vehicles[self.current_vehicle_idx]
        total = len(vehicle["frames"])
        if total <= 0:
            return

        next_idx = self.current_vehicle_frame_idx + direction
        if next_idx >= total:
            if loop:
                next_idx = 0
            else:
                next_idx = total - 1
                if self.is_crop_playing:
                    self.toggle_crop_play()
        elif next_idx < 0:
            next_idx = 0

        self.current_vehicle_frame_idx = next_idx
        self.update_vehicle_crop_display()

    def toggle_play(self):
        if self.video_cap is None:
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            interval_ms = max(1, int(1000 / max(self.fps, 1.0)))
            self.timer.start(interval_ms)
        else:
            self.btn_play.setText("Play / Pause")
            self.timer.stop()

    def step_frame(self, direction, loop=False):
        if self.video_cap is None:
            return

        next_idx = self.current_frame_idx + direction
        if next_idx >= self.total_frames:
            if loop:
                next_idx = 0
            else:
                next_idx = max(0, self.total_frames - 1)
                if self.is_playing:
                    self.toggle_play()
        elif next_idx < 0:
            next_idx = 0

        self.show_frame(next_idx)

    def stop_playback(self):
        if self.is_playing:
            self.toggle_play()
        if self.is_crop_playing:
            self.toggle_crop_play()

    def release_video_cap(self):
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def closeEvent(self, event):
        self.stop_playback()
        self.release_video_cap()
        if self.recog_thread is not None and self.recog_thread.isRunning():
            self.recog_thread.stop()
            self.recog_thread.wait(2000)
        if self.video_recog_thread is not None and self.video_recog_thread.isRunning():
            self.video_recog_thread.stop()
            self.video_recog_thread.wait(2000)
        if self.head_tail_recog_thread is not None and self.head_tail_recog_thread.isRunning():
            self.head_tail_recog_thread.stop()
            self.head_tail_recog_thread.wait(2000)
        if self.tracker_thread is not None and self.tracker_thread.isRunning():
            self.tracker_thread.stop()
            self.tracker_thread.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())
