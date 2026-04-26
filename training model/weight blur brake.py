import os, sys, json, yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

project_path = '/kaggle/input/datasets/taddyz6/tld-project-zjh/TLD_base_zjh'
data_root    = '/kaggle/input/datasets/taddyz6/tld-traffic-data'
output_dir   = '/kaggle/working/output'
config_path  = '/kaggle/working/clean_config.yaml'

os.makedirs(output_dir, exist_ok=True)
if project_path not in sys.path:
    sys.path.append(project_path)
os.chdir(project_path)


config_content = {
    'work_dir': output_dir,
    'random_fix': True,
    'random_seed': 123,
    'device': '0',
    'num_worker': 4,
    'batch_size': 32,
    'test_batch_size': 16,
    'num_epoch': 10,
    'save_interval': 1,
    'eval_interval': 1,
    'print_log': True,
    'log_interval': 100,
    'model': 'TLD_resnet',
    'feeder': 'VideoFeeder',
    'dataset': 'TLD_YT',
    'model_args': {
        'loss_weights': {'turn': 1.0, 'brake': 1.0}
    },
    'optimizer_args': {
        'optimizer': 'AdamW',
        'base_lr': 0.001,
        'learning_ratio': 1,
        'step': [5, 7],
        'weight_decay': 0.0001,
        'start_epoch': 0,
        'nesterov': False
    }
}
with open(config_path, 'w') as f:
    yaml.dump(config_content, f)

import datasets.video_feeder as vf

def resolve_image_path(path: str) -> str:
    p = path.lstrip('/')
    if p.startswith('TLD/'):  
        p = p[len('TLD/'):]
    cand = os.path.join(data_root, p)
    if os.path.exists(cand):
        return cand
    if "TLD/" in p:
        p2 = p[p.index("TLD/"):]
        if p2.startswith('TLD/'):
            p2 = p2[len('TLD/'):]
        cand2 = os.path.join(data_root, p2)
        if os.path.exists(cand2):
            return cand2
    raise FileNotFoundError(f"Image not found. orig={path} | tried={cand}")

def safe_pil_loader(path: str) -> np.ndarray:
    final_path = resolve_image_path(path)
    with Image.open(final_path) as img:
        return np.array(img.convert('RGB'))

def patched_read_video(self, index):
    info_video = self.inputs_list[index]
    img_path = info_video.get('file_name') or info_video.get('img_path')
    data = safe_pil_loader(img_path)
    return (data, info_video.get('car_label', []), info_video.get('car_num', 0), info_video)

def patched_normalize_and_crop(self, video, label, car_num):
    video_tensor = torch.from_numpy(video).permute(2, 0, 1).float() 

    if car_num <= 0 or not label:
        return [], []

    if not isinstance(label, list):
        label = [label]

    full_video, full_label = [], []
    turn_map = {'off': 0, 'left': 1, 'right': 2, 'both': 3, 'unknow': 4}

    blur_p = 0.3      
    blur_kernel = 3    
    blur_sigma = (0.1, 2.0) 
    blur_tf = transforms.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)

    for item in label:
        try:
            coord = np.array(item['bounding_boxes']['coordinate'])
            x1, y1 = int(coord.min(0)[0]), int(coord.min(0)[1])
            x2, y2 = int(coord.max(0)[0]), int(coord.max(0)[1])

            h, w = video_tensor.shape[1], video_tensor.shape[2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x1 >= x2 or y1 >= y2:
                continue

            cropped = video_tensor[:, y1:y2, x1:x2]
            resized = transforms.Resize((224, 224))(cropped)
            resized = resized.permute(1, 2, 0).contiguous()

            brake = 0 if item.get('brake_label') == 'car_BrakeOff' else 1
            turn  = turn_map.get(item.get('turn_label'), 4)

            full_video.append(resized)
            full_label.append(torch.LongTensor([brake, turn]))
        except:
            continue

    if len(full_video) == 0:
        return [], []

    full_video = [v / 127.5 - 1.0 for v in full_video]
    return full_video, full_label

def patched_init(self, mode, prefix=None):
    self.mode = mode
    self.prefix = data_root
    label_path = os.path.join(project_path, 'datasets', f'TLD_YT_{mode}.json')
    if not os.path.exists(label_path):
        label_path = os.path.join(project_path, f'TLD_YT_{mode}.json')
    with open(label_path, 'r') as f:
        self.inputs_list = json.load(f)
    print(f"✅ {mode} 模式加载成功，数据量: {len(self.inputs_list)}")

def patched_collate_fn(batch):
    car_num_json = 0
    for b in batch:
        try:
            car_num_json += int(b[2])
        except:
            pass

    batch2 = [b for b in batch if len(b[0]) > 0 and len(b[1]) > 0]

    if len(batch2) == 0:
        return {
            'x': torch.empty((0, 224, 224, 3), dtype=torch.float32),
            'label': torch.empty((0, 2), dtype=torch.long),
            'car_num': 0,
            'car_num_roi': 0,
            'car_num_json': car_num_json,
            'origin_info': tuple(b[3] for b in batch),
        }

    batch2 = sorted(batch2, key=lambda x: len(x[0]), reverse=True)
    video, label, car_num, info = list(zip(*batch2))

    video = torch.stack([img for sub in video for img in sub])
    label = torch.stack([lab for sub in label for lab in sub])
    car_num_roi = int(label.shape[0])

    return {
        'x': video,
        'label': label,
        'car_num': car_num_roi,
        'car_num_roi': car_num_roi,
        'car_num_json': car_num_json,
        'origin_info': info,
    }

vf.VideoFeeder.__init__ = patched_init
vf.VideoFeeder.read_video = patched_read_video
vf.VideoFeeder.normalize_and_crop = patched_normalize_and_crop
vf.VideoFeeder.collate_fn = staticmethod(patched_collate_fn)

import seq_scripts
from tqdm import tqdm

def seq_eval_patched(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    model.eval()
    turn_correct = 0
    brake_correct = 0
    turn_sum_roi = 0
    brake_sum_roi = 0
    turn_sum_json = 0
    brake_sum_json = 0

    with torch.no_grad():
        for _, data in enumerate(tqdm(loader)):
            data = device.dict_data_to_device(data)

            if data.get('car_num_roi', 0) == 0 or data['label'].numel() == 0:
                turn_sum_json += int(data.get('car_num_json', 0))
                brake_sum_json += int(data.get('car_num_json', 0))
                continue

            labels_turn = data['label'][:, 1]
            labels_brake = data['label'][:, 0]

            ret_dict = model(data)
            _, predicted_turn = torch.max(ret_dict['turn_result'], 1)
            _, predicted_brake = torch.max(ret_dict['brake_result'], 1)

            turn_correct += (predicted_turn == labels_turn).sum().item()
            brake_correct += (predicted_brake == labels_brake).sum().item()

            denom_roi = int(data.get('car_num_roi', data.get('car_num', 0)))
            denom_json = int(data.get('car_num_json', 0))
            turn_sum_roi += denom_roi
            brake_sum_roi += denom_roi
            turn_sum_json += denom_json
            brake_sum_json += denom_json

    roi_turn_acc = (turn_correct / turn_sum_roi) if turn_sum_roi > 0 else 0.0
    roi_brake_acc = (brake_correct / brake_sum_roi) if brake_sum_roi > 0 else 0.0
    json_turn_acc = (turn_correct / turn_sum_json) if turn_sum_json > 0 else 0.0
    json_brake_acc = (brake_correct / brake_sum_json) if brake_sum_json > 0 else 0.0

    msg = (
        f'\tEpoch: {epoch} | '
        f'ROI acc: Turn {roi_turn_acc*100:.2f}% Brake {roi_brake_acc*100:.2f}% | '
        f'JSON acc: Turn {json_turn_acc*100:.2f}% Brake {json_brake_acc*100:.2f}% | '
        f'correct: turn={turn_correct} brake={brake_correct} | '
        f'sums: roi={turn_sum_roi} json={turn_sum_json}'
    )
    recoder.print_log(msg)
    recoder.print_log(msg, path=f'{work_dir}test.txt')
    return 0

seq_scripts.seq_eval = seq_eval_patched

import main

def judge_save_eval_every_epoch(self, epoch):
    return True, True
main.SLRProcessor.judge_save_eval = judge_save_eval_every_epoch

if not hasattr(main.SLRProcessor, "_orig_save_model__safe"):
    main.SLRProcessor._orig_save_model__safe = main.SLRProcessor.save_model

def save_model_safe(self, epoch, model_path=None):
    orig = main.SLRProcessor._orig_save_model__safe
    try:
        if model_path is not None:
            orig(self, epoch, model_path)
        else:
            orig(self, epoch)
    except TypeError:
        orig(self, epoch)

    ckpt_epoch = os.path.join(self.arg.work_dir, f'cur_test_model_epoch{epoch}.pt')
    ckpt_latest = os.path.join(self.arg.work_dir, 'cur_test_model_latest.pt')
    torch.save(self.model.state_dict(), ckpt_epoch)
    torch.save(self.model.state_dict(), ckpt_latest)

    if getattr(self.arg, "print_log", False):
        msg1 = f"\t[Save] {ckpt_epoch}"
        msg2 = f"\t[Save] {ckpt_latest}"
        if hasattr(self, "recoder") and hasattr(self.recoder, "print_log"):
            self.recoder.print_log(msg1)
            self.recoder.print_log(msg2)
        else:
            print(msg1)
            print(msg2)

main.SLRProcessor.save_model = save_model_safe

import light_network

if not hasattr(light_network.TLD_resnet, "_orig_get_loss__brake_weighted"):
    light_network.TLD_resnet._orig_get_loss__brake_weighted = light_network.TLD_resnet.get_loss

def get_loss_brake_weighted(self, out, label):
    device = out['brake_result'].device

    w = torch.tensor([1.0, 2.0], device=device)
    brake_loss_fn = nn.CrossEntropyLoss(weight=w)
    turn_loss_fn  = nn.CrossEntropyLoss()

    brake = brake_loss_fn(out['brake_result'], label[:, 0])
    turn  = turn_loss_fn(out['turn_result'],  label[:, 1])

    lw = getattr(self, "loss_weights", {'turn': 1.0, 'brake': 1.0})
    total = lw.get('brake', 1.0) * brake + lw.get('turn', 1.0) * turn

    loss_details = {
        'brake_loss': brake.detach(),
        'turn_loss': turn.detach(),
    }
    return total, loss_details

light_network.TLD_resnet.get_loss = get_loss_brake_weighted
print("[OK] Patched get_loss(): brake uses weighted CE and returns (loss, loss_details).")

train_json = os.path.join(project_path, 'datasets', 'TLD_YT_train.json')
if os.path.exists(train_json):
    lst = json.load(open(train_json, 'r'))
    z = sum(1 for x in lst if x.get('car_num', 0) == 0)
    print(f"[Sanity] car_num==0: {z}/{len(lst)} = {z/len(lst):.2%}")


from main import SLRProcessor
import utils

sys.argv = ['main.py', '--config', config_path, '--work-dir', output_dir, '--device', '0']
parser = utils.get_parser()
args = parser.parse_args()

with open(config_path, 'r') as f:
    default_arg = yaml.safe_load(f)
parser.set_defaults(**default_arg)
args = parser.parse_args()

if torch.cuda.is_available():
    print(f"🚀 使用 GPU: {torch.cuda.get_device_name(0)} 启动训练...")
else:
    print("⚠️ 未检测到 GPU，请检查右侧 Settings 开启 Accelerator。")



import os, glob
import torch
import light_network
from torchvision.models.resnet import resnet18 as resnet18_builder

WEIGHT_DIR = "/kaggle/input/datasets/taddyz6/resnet18-weight"

cands = []
cands += glob.glob(os.path.join(WEIGHT_DIR, "**", "*.pth"), recursive=True)
cands += glob.glob(os.path.join(WEIGHT_DIR, "**", "*.pt"),  recursive=True)
if len(cands) == 0:
    raise FileNotFoundError(f"No .pth/.pt found under: {WEIGHT_DIR}")

cands_sorted = sorted(
    cands,
    key=lambda p: (("resnet18" not in os.path.basename(p).lower()), len(p))
)
WEIGHT_PATH = cands_sorted[0]
print("[OK] Using local resnet18 weights:", WEIGHT_PATH)

if not hasattr(light_network, "_orig_resnet18_builder"):
    light_network._orig_resnet18_builder = resnet18_builder

def resnet18_local_pretrained(*args, **kwargs):
    kwargs.pop("pretrained", None)
    kwargs.pop("weights", None)

    m = light_network._orig_resnet18_builder(weights=None, **kwargs)

    sd = torch.load(WEIGHT_PATH, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    if isinstance(sd, dict):
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[OK] Loaded local weights. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 10):", missing[:10])
    if unexpected:
        print("  unexpected (first 10):", unexpected[:10])

    return m

light_network.models.resnet18 = resnet18_local_pretrained
print("[OK] Patched light_network.models.resnet18 to load local weights (no internet).")






processor = SLRProcessor(args)

it = iter(processor.data_loader['test'])
d = next(it)
print("[Sanity] test batch x shape:", d['x'].shape)
print("[Sanity] test batch label shape:", d['label'].shape)
print("[Sanity] car_num_roi:", d.get('car_num_roi'))
print("[Sanity] car_num_json:", d.get('car_num_json'))

processor.start()

print("\n[Output] saved to:", output_dir)
print("Expect: cur_test_model_epoch0.pt ... and cur_test_model_latest.pt")