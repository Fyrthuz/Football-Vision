from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    task="detect",
    data="/app/datasets/players/data.yaml",
    epochs=100,
    patience=30,
    batch=6,
    imgsz=1280,
    cache=True,
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    close_mosaic=10,
    pretrained=True,
    amp=True,
    plots=True,
    val=True,
    project="models",
    name="players_m",
    exist_ok=True,
)
