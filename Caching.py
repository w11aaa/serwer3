import hashlib as Hashlib
import os as OS
import torch as Torch
from ultralytics import YOLO

try:
    import streamlit as App
except ImportError:
    class MockApp:
        def cache_resource(self, *args, **kwargs):
            return lambda x: x
        def toast(self, *args, **kwargs):
            pass
        def error(self, msg):
            print(f"ERROR: {msg}")
    App = MockApp()

@App.cache_resource(show_spinner=False)
def BuildModel(ModelPath, Extension, device_preference='auto'):
    if Extension.upper() == ".ENGINE" and not Torch.cuda.is_available():
        App.toast("TensorRT只能在NVIDIA硬件上运行", icon="❌")
        return None
    
    # 检查设备偏好设置
    if Extension.upper() == ".ENGINE" and device_preference == 'cpu':
        App.toast("TensorRT模型无法在CPU上运行，将自动使用GPU", icon="⚠️")
        device_preference = 'cuda'
    
    # 对于 ONNX 模型，如果安装了 onnxruntime-gpu 则优先使用 GPU
    if Extension.upper() == ".ONNX":
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in providers and Torch.cuda.is_available() and device_preference != 'cpu':
                App.toast("检测到 ONNX GPU 加速环境，已启用 CUDA 推理", icon="🚀")
            else:
                App.toast("ONNX 模型将运行在 CPU 模式", icon="ℹ️")
        except ImportError:
            App.toast("未检测到 onnxruntime，将使用默认后端", icon="⚠️")

    # 直接从路径加载，不再写入临时文件
    # 创建模型
    try:
        NewM = YOLO(ModelPath, task="segment")
    except Exception as e:
        App.error(f"模型加载失败 ({ModelPath}): {str(e)}")
        return None
    
    # 根据设备偏好设置模型设备
    if device_preference == 'cpu':
        NewM.to('cpu')
    elif device_preference == 'cuda' and Torch.cuda.is_available():
        NewM.to('cuda')
    elif device_preference == 'auto':
        # 自动选择：优先GPU，不可用则CPU
        if Torch.cuda.is_available():
            NewM.to('cuda')
        else:
            NewM.to('cpu')
    
    # 预热模型
    try:
        GetSample = (1, 3, 640, 640)
        # 对于不同类型的模型，获取设备的方式不同
        if hasattr(NewM, 'model') and hasattr(NewM.model, 'parameters'):
            device = next(NewM.model.parameters()).device
            NewM(Torch.zeros(GetSample).to(device))
        else:
            # 对于 ONNX/TensorRT 等导出模型，直接进行一次空推理
            import numpy as np
            dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
            NewM(dummy_input)
    except Exception as e:
        App.toast(f"模型预热跳过: {str(e)}", icon="ℹ️")
    
    return NewM

def ExportModel(ModelPath, TargetFormat="onnx"):
    """导出模型为指定格式 (onnx, engine 等)"""
    try:
        model = YOLO(ModelPath)
        # 导出模型
        exported_path = model.export(format=TargetFormat)
        return exported_path
    except Exception as e:
        App.error(f"模型导出失败: {str(e)}")
        return None