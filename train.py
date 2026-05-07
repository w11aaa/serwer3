import torch
from ultralytics import YOLO
import traceback

if __name__ == '__main__':
    # 定义所有要跑的实验配置
    # 格式：(模型权重/配置文件, 实验名称, batch_size)
    # 注意：模型越大，batch_size 需要相应减小以防止显存溢出 (OOM)
    experiments = [
        # --- 横向对比实验 (配角) ---
        ('yolov5n-seg.pt', 'yolov5n_baseline', 8),
        # ('yolov8s-seg.pt',  'yolov8s_baseline', 8),
        # ('yolov8m-seg.pt',  'yolov8m_baseline', 4),  # m模型较大，batch降为4
        # ('yolo11n-seg.pt',  'yolo11n_baseline', 8),
        
        # --- 如果你之前还没跑完原版的 v8n，可以把下面这行取消注释 ---
        # ('yolov8n-seg.pt',  'yolov8n_baseline', 8),
        
        # --- 如果你想再跑一次你的轻量化网络，可以把下面这行取消注释 ---
        # ('yolov8n-ghost-seg.yaml', 'yolov8n_ghost_exp2', 8)
    ]

    # 公共参数配置
    DATA_YAML = 'F:/Sewer2/mode_train/data_seg/data_seg.yaml'
    EPOCHS = 50
    IMGSZ = 640
    PROJECT_DIR = 'runs/segment'

    print(f"🚀 准备开始批量训练，共计 {len(experiments)} 个模型待运行...\n")

    for i, (model_path, exp_name, batch_size) in enumerate(experiments):
        print("=" * 60)
        print(f"🌟 开始训练第 {i+1}/{len(experiments)} 个模型: {exp_name}")
        print(f"📦 加载权重/配置: {model_path} | Batch Size: {batch_size}")
        print("=" * 60)

        try:
            # 1. 初始化模型 (会自动下载缺失的 .pt 官方权重)
            model = YOLO(model_path)

            # 2. 开始训练
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=batch_size,
                project=PROJECT_DIR,
                name=exp_name,
                # 下面这两个参数可以稍微加速训练
                cache=False,  # 如果内存大可以设为 True，8GB内存建议 False
                workers=8     # 保持多线程读取
            )
            
            print(f"✅ 模型 {exp_name} 训练圆满完成！\n")

        except Exception as e:
            # 如果报错了，打印错误信息，但不要停止整个脚本，继续跑下一个模型
            print(f"❌ 模型 {exp_name} 训练失败！")
            print("错误信息如下：")
            traceback.print_exc()
            print("\n跳过该模型，继续执行下一个...\n")

        finally:
            # 3. 释放显存，防止上一个模型的残留导致下一个模型 OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 已清理 GPU 显存，准备迎接下一个任务...\n")

    print("🎉 所有自动化实验已全部运行完毕！赶紧去 results.csv 里收割数据吧！")