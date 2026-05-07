import os
import time
import torch
import pandas as pd
from ultralytics import YOLO

# ⚠️ 请确保这里的路径正确
runs_dir = r"F:\Sewer2\mode_train\runs\segment\runs\segment"
test_dir = r"F:\Sewer2\mode_train\data_seg\valid\images"

# 你所有的 8 个模型
models = [
    "yolov8m_baseline", "yolov8s_baseline", "yolov8n_baseline", "yolo11n_baseline",
    "yolo11_c2f_exp", "yolov8n_ghost_exp", "yolo11_dw_exp", "yolo11n_ghost_exp"
]

# 获取测试集所有图片
image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
num_images = len(image_paths)

print(f"📁 找到测试图片: {num_images} 张")
print(f"{'模型名称':<20} | {'Mask mAP50':<12} | {'参数量(M)':<10} | {'权重大小(MB)':<12} | {'平均耗时(ms)':<10}")
print("-" * 75)

for exp in models:
    exp_dir = os.path.join(runs_dir, exp)
    csv_file = os.path.join(exp_dir, "results.csv")
    weight_path = os.path.join(exp_dir, "weights", "best.pt")
    
    mask_map = 0.0
    total_params = 0.0
    size_mb = 0.0
    avg_time_ms = 0.0
    
    try:
        # ==========================================
        # 1. 提取精度 (从 results.csv 中读取)
        # ==========================================
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip() # 清理空格
            mask_map = df['metrics/mAP50(M)'].tail(100).max() * 100
            
        # ==========================================
        # 2. 提取体积与测速 (加载 best.pt)
        # ==========================================
        if os.path.exists(weight_path) and num_images > 0:
            size_mb = os.path.getsize(weight_path) / (1024 * 1024)
            model = YOLO(weight_path)
            total_params = sum(p.numel() for p in model.model.parameters()) / 1e6
            
            # 预热显卡
            for _ in range(3):
                model.predict(image_paths[0], verbose=False)
                
            # 批量测试计算平均耗时
            start_time = time.time()
            for img_path in image_paths:
                model.predict(img_path, verbose=False)
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) * 1000) / num_images
            
            # 清理显存防溢出
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # ==========================================
        # 3. 打印终极报表格式
        # ==========================================
        print(f"{exp:<20} | {mask_map:>9.2f}% | {total_params:>8.2f} M | {size_mb:>9.2f} MB | {avg_time_ms:>8.2f} ms")
        
    except Exception as e:
        print(f"{exp:<20} | ❌ 读取或测试失败: {e}")

print("-" * 75)
print("🎉 恭喜！论文核心实验数据已全部生成完毕！")