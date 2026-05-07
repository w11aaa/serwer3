from ultralytics import YOLO

# 请确保已安装 ultralytics，并且当前目录为工作区根目录
# 通过以下命令运行：
# python train_yolov8_seg.py
if __name__ == '__main__':
    model = YOLO('yolov8n-ghost-seg.yaml')
    model.train(
        data='F:/Sewer2/mode_train/data_seg/data_seg.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        project='runs/segment',
        name='yolov8n_ghost_exp'
    )
