import torch
from ultralytics import YOLO
import traceback

if __name__ == '__main__':
    # ==========================================
    # 你的 YOLOv11 终极消融实验配置清单
    # 格式：(你写的 yaml 配置文件名, 保存的文件夹名称, batch_size)
    # ==========================================
    experiments = [
        # 1. 极致参数压缩 (跨代缝合 C2fGhost)
        ('yolo11n-c2fghost-seg.yaml', 'yolo11_c2fghost_exp', 8),
        
        # 2. 精度补偿 (Ghost + CBAM 注意力)
        ('yolo11n-ghost-cbam-seg.yaml', 'yolo11_ghost_cbam_exp', 8),
        
        # 3. 空间解耦 (DWConv 深度可分离卷积)
        ('yolo11n-dw-seg.yaml', 'yolo11_dw_exp', 8),
        
        # 4. 非对称网络 (强主干 + 轻颈部 LiteNeck)
        ('yolo11n-liteneck-seg.yaml', 'yolo11_liteneck_exp', 8),
        
        # 5. 架构退化对照组 (退化为纯 C2f)
        ('yolo11n-c2f-seg.yaml', 'yolo11_c2f_exp', 8)
    ]

    # 全局公共参数（千万别动，保证控制变量绝对公平）
    DATA_YAML = 'F:/Sewer2/mode_train/data_seg/data_seg.yaml'
    EPOCHS = 50
    IMGSZ = 640
    PROJECT_DIR = 'runs/segment'

    print(f"🚀 [毕业论文核心战役] 准备开始批量训练，共计 {len(experiments)} 个高阶消融模型待运行...\n")

    for i, (yaml_file, exp_name, batch_size) in enumerate(experiments):
        print("=" * 70)
        print(f"🌟 开始训练第 {i+1}/{len(experiments)} 个模型: {exp_name}")
        print(f"📦 加载架构: {yaml_file} | Batch Size: {batch_size}")
        print("=" * 70)

        try:
            # 1. 根据你的 yaml 文件初始化模型架构
            # ⚠️ 注意：这里传入的是 .yaml 而不是 .pt，意味着模型将从零开始(随机权重)学习你的网络结构
            model = YOLO(yaml_file)

            # 2. 开始高强度炼丹
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=batch_size,
                project=PROJECT_DIR,
                name=exp_name,
                cache=False,  # 8GB显存建议关掉，防爆显存
                workers=8,    # 开启8线程加速数据读取
                amp=True      # 开启自动混合精度加速
            )
            
            print(f"\n✅ 恭喜！模型 {exp_name} 训练圆满完成！\n")

        except Exception as e:
            # 万一你某个 yaml 文件缩进写错了导致报错，这里会接住它，不会让整个程序崩溃
            print(f"\n❌ 模型 {exp_name} 训练遭遇滑铁卢！")
            print("错误现场记录：")
            traceback.print_exc()
            print("\n没关系，果断跳过，继续执行下一个...\n")

        finally:
            # 3. 护航机制：每个模型跑完，强制吸尘器清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU 显存已强制清空，为下一个模型腾出跑道...\n")

    print("🎉 漫长的一夜结束了，所有的消融实验均已运行完毕！赶紧用 get_results.py 去提取论文数据吧！")