from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = Path(r"F:\18744\vehicle_dataset\data.yaml")

    # Fine-tune from your trained model
    model = YOLO(r"F:\18744\best.pt")

    results = model.train(
        data=str(data_yaml),

        # Core
        epochs=60,              # with 100k imgs, 60 is already a lot
        imgsz=640,
        batch=-1,               # auto batch; if unstable/OOM, set 8 or 4
        device=0,
        workers=4,
        amp=True,

        # Speed on big data (Windows)
        cache="disk",           # big speedup if you have SSD space
        rect=True,              # often improves speed/memory, stable for detection

        # Augs (good defaults, don’t go crazy)
        close_mosaic=10,
        mosaic=0.8,             # keep some mosaic early
        mixup=0.0,              # usually not needed for vehicle-only detection
        fliplr=0.5,
        scale=0.5,

        # Training behavior
        lr0=0.003,              # lower LR for fine-tuning (safer than default)
        patience=20,            # don’t wait forever with 100k imgs
        warmup_epochs=2,

        project="runs_detect",
        name="ft_bestpt_640",
        exist_ok=True,
    )

    print("Best weights:", results.save_dir / "weights" / "best.pt")

if __name__ == "__main__":
    main()
