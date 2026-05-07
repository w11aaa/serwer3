import os
import pandas as pd

# 你刚才 dir 命令显示的路径
runs_dir = r"F:\Sewer2\mode_train\runs\segment\runs\segment"

print(f"{'模型名称':<20} | {'Mask mAP50 (%)':<15} | {'Box mAP50 (%)':<15}")
print("-" * 55)

# 遍历目录下的所有实验文件夹
for exp_name in os.listdir(runs_dir):
    exp_path = os.path.join(runs_dir, exp_name)
    csv_file = os.path.join(exp_path, "results.csv")
    
    if os.path.isdir(exp_path) and os.path.exists(csv_file):
        try:
            # 读取训练日志
            df = pd.read_csv(csv_file)
            # 清理列名空格
            df.columns = df.columns.str.strip()
            
            # YOLOv8/11 的 mAP50 列名通常是 metrics/mAP50(M) 和 metrics/mAP50(B)
            # 取最后 5 轮的最高值作为最终成绩
            mask_map = df['metrics/mAP50(M)'].tail(5).max() * 100
            box_map = df['metrics/mAP50(B)'].tail(5).max() * 100
            
            print(f"{exp_name:<20} | {mask_map:>12.2f}% | {box_map:>12.2f}%")
        except Exception as e:
            print(f"{exp_name:<20} | 读取数据出错")