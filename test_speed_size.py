from ultralytics import YOLO
import os
import torch
import time

runs_dir = r"F:\Sewer2\mode_train\runs\segment\runs\segment"
# ⚠️ 将这里修改为你的验证集文件夹路径
test_dir = r"F:\Sewer2\mode_train\data_seg\test\images" 

models = [
    "yolo11n_baseline", "yolov8m_baseline", "yolov8s_baseline", "yolov8n_baseline",
    "yolo11_c2f_exp", "yolov8n_ghost_exp", "yolo11_dw_exp", "yolo11n_ghost_exp"
]

# 获取文件夹下所有的 jpg/png 图片路径
image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
num_images = len(image_paths)

print(f"📁 找到测试图片: {num_images} 张")
print(f"{'模型名称':<20} | {'参数量(M)':<10} | {'权重大小(MB)':<12} | {'平均耗时(ms/图)':<15}")
print("-" * 75)

for exp in models:
    weight_path = os.path.join(runs_dir, exp, "weights", "best.pt")
    if os.path.exists(weight_path) and num_images > 0:
        model = YOLO(weight_path)
        
        # 1. 获取物理大小与参数量
        size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        total_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        
        try:
            # 2. 显卡预热 (Warm-up) - 拿第一张图连续跑3次，不计入时间
            for _ in range(3):
                model.predict(image_paths[0], verbose=False)
            
            # 3. 正式测速循环
            start_time = time.time()
            for img_path in image_paths:
                model.predict(img_path, verbose=False)
            end_time = time.time()
            
            # 4. 计算平均耗时
            total_time_ms = (end_time - start_time) * 1000
            avg_time_ms = total_time_ms / num_images
            
        except Exception as e:
            print(f"测速报错: {e}")
            avg_time_ms = 0.0
            
        print(f"{exp:<20} | {total_params:>8.2f} M | {size_mb:>9.2f} MB | {avg_time_ms:>10.2f} ms")
        
        # 释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()