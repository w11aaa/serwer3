from ultralytics import YOLO

if __name__ == '__main__':
    # ⚠️ 注意这里：加载的是你刚刚新建的 yolo11 yaml 文件
    # 模型会随机初始化，从头开始学习你的下水道数据
    model = YOLO('yolo11n-ghost-seg.yaml') 

    model.train(
        data='F:/Sewer2/mode_train/data_seg/data_seg.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        project='runs/segment',
        name='yolo11n_ghost_exp'
    )