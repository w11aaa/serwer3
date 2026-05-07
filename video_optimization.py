"""
视频分析性能优化模块
提供多种优化策略来提升视频处理速度
"""

import cv2 as CV2
import numpy as NP
import torch as Torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from typing import List, Dict, Any, Tuple
import multiprocessing as mp

class VideoOptimizer:
    """视频分析优化器"""
    
    def __init__(self, model, batch_size=8, num_workers=4, use_gpu=True):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu and Torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'processing_time': 0,
            'fps': 0
        }
    
    def preprocess_frame(self, frame: NP.ndarray, target_size: Tuple[int, int] = (640, 640)) -> NP.ndarray:
        """预处理帧 - 优化版本"""
        # 如果帧已经很小，跳过resize
        h, w = frame.shape[:2]
        if h <= target_size[0] and w <= target_size[1]:
            return frame
        
        # 保持宽高比的resize
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 使用更快的插值方法
        resized = CV2.resize(frame, (new_w, new_h), interpolation=CV2.INTER_LINEAR)
        
        # 如果尺寸不匹配，进行padding
        if new_h != target_size[0] or new_w != target_size[1]:
            pad_h = (target_size[0] - new_h) // 2
            pad_w = (target_size[1] - new_w) // 2
            resized = CV2.copyMakeBorder(resized, pad_h, target_size[0] - new_h - pad_h,
                                       pad_w, target_size[1] - new_w - pad_w, CV2.BORDER_CONSTANT)
        
        return resized
    
    def batch_inference(self, frames: List[NP.ndarray]) -> List[Any]:
        """批量推理 - 显著提升GPU利用率"""
        if not frames:
            return []
        
        # 预处理所有帧
        processed_frames = [self.preprocess_frame(frame) for frame in frames]
        
        # 检查是否使用半精度
        model_to_check = self.model.model if hasattr(self.model, 'model') else self.model
        is_fp16 = False
        try:
            is_fp16 = next(model_to_check.parameters()).dtype == Torch.float16
        except Exception:
            pass
            
        if self.use_gpu:
            # 直接使用Ultralytics的批量推理，传递numpy数组列表
            # 它会自动处理预处理、归一化和半精度转换
            # 关键是传递 half=True (如果模型是FP16) 和 device
            results = self.model(
                processed_frames, 
                half=is_fp16, 
                device=self.device, 
                verbose=False
            )
            return results
        else:
            # CPU模式
            results = self.model(
                processed_frames, 
                device='cpu', 
                verbose=False
            )
            return results

    def parallel_analysis(self, results: List[Any]) -> List[Dict[str, Any]]:
        """并行分析 - 利用多核CPU"""
        from Analysis import AnalyzePicture, ShowMask
        
        def analyze_single(result):
            try:
                GetImgLikes, GetResult = AnalyzePicture(result)
                Plot, Mask, Ellipse = GetImgLikes
                
                Record = {
                    "AspectRatio": GetResult["AspectRatio"][0] if GetResult["AspectRatio"] else None,
                    "Orientation": GetResult["Orientation"][0] if GetResult["Orientation"] else None,
                    "Deformation": GetResult["Deformation"][0] if GetResult["Deformation"] else None,
                    "Plot": Plot,
                    "Mask": Mask,
                    "Shape": GetResult["Shape"][0] if GetResult["Shape"] else "Undefined"
                }
                
                # 只在需要时生成Draw图像
                if Ellipse is not None:
                    Draw = ShowMask(Mask, Ellipse, Record["Shape"])
                    Record["Draw"] = Draw
                else:
                    Record["Draw"] = Mask
                
                return Record
            except Exception as e:
                print(f"分析错误: {e}")
                return None
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            analyzed_results = list(executor.map(analyze_single, results))
        
        # 过滤掉None结果
        return [r for r in analyzed_results if r is not None]
    
    def optimized_video_processing(self, video_path: str, frame_spacing: int = 1) -> List[Dict[str, Any]]:
        """优化的视频处理流程"""
        start_time = time.time()
        
        # 打开视频
        cap = CV2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(CV2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(CV2.CAP_PROP_FPS)
        
        self.stats['total_frames'] = total_frames
        
        all_results = []
        frame_batch = []
        frame_count = 0
        
        print(f"开始处理视频: {total_frames} 帧, FPS: {fps:.2f}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按帧间距采样
            if frame_count % frame_spacing == 0:
                frame_batch.append(frame)
                
                # 当批次满了或到达最后一帧时处理
                if len(frame_batch) >= self.batch_size:
                    # 批量推理
                    results = self.batch_inference(frame_batch)
                    
                    # 并行分析
                    analyzed = self.parallel_analysis(results)
                    all_results.extend(analyzed)
                    
                    self.stats['processed_frames'] += len(frame_batch)
                    
                    # 清空批次
                    frame_batch = []
                    
                    # 显示进度
                    progress = self.stats['processed_frames'] / total_frames * 100
                    print(f"处理进度: {progress:.1f}% ({self.stats['processed_frames']}/{total_frames})")
            
            frame_count += 1
        
        # 处理剩余的帧
        if frame_batch:
            results = self.batch_inference(frame_batch)
            analyzed = self.parallel_analysis(results)
            all_results.extend(analyzed)
            self.stats['processed_frames'] += len(frame_batch)
        
        cap.release()
        
        # 计算性能统计
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.stats['fps'] = self.stats['processed_frames'] / processing_time if processing_time > 0 else 0
        
        print(f"处理完成: {self.stats['processed_frames']} 帧, 耗时: {processing_time:.2f}s, 平均FPS: {self.stats['fps']:.2f}")
        
        return all_results

class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    def optimize_model_memory(model):
        """优化模型内存使用"""
        # 获取实际的PyTorch模型
        real_model = model.model if hasattr(model, 'model') else model
        
        if Torch.cuda.is_available():
            # 启用混合精度
            real_model.half()  # 使用FP16
            Torch.backends.cudnn.benchmark = True
            Torch.backends.cudnn.deterministic = False
        
        # 设置推理模式
        # 注意：YOLO包装器没有eval()方法，但其内部模型有
        if hasattr(model, 'eval'):
            model.eval()
        else:
            real_model.eval()
        
        return model
    
    @staticmethod
    def clear_gpu_cache():
        """清理GPU缓存"""
        if Torch.cuda.is_available():
            Torch.cuda.empty_cache()
            Torch.cuda.synchronize()

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_monitoring(self, name: str):
        """开始监控"""
        self.metrics[name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
    
    def end_monitoring(self, name: str):
        """结束监控"""
        if name in self.metrics:
            self.metrics[name]['end_time'] = time.time()
            self.metrics[name]['duration'] = (
                self.metrics[name]['end_time'] - self.metrics[name]['start_time']
            )
    
    def get_report(self) -> Dict[str, float]:
        """获取性能报告"""
        return {name: data['duration'] for name, data in self.metrics.items() 
                if data['duration'] is not None}

# 优化建议和配置
OPTIMIZATION_CONFIGS = {
    'high_performance': {
        'batch_size': 16,
        'num_workers': mp.cpu_count(),
        'use_gpu': True,
        'frame_spacing': 1,
        'description': '高性能模式 - 最大处理速度'
    },
    'balanced': {
        'batch_size': 8,
        'num_workers': max(2, mp.cpu_count() // 2),
        'use_gpu': True,
        'frame_spacing': 2,
        'description': '平衡模式 - 速度与质量平衡'
    },
    'memory_efficient': {
        'batch_size': 4,
        'num_workers': 2,
        'use_gpu': True,
        'frame_spacing': 5,
        'description': '内存优化模式 - 低内存使用'
    },
    'cpu_only': {
        'batch_size': 2,
        'num_workers': mp.cpu_count(),
        'use_gpu': False,
        'frame_spacing': 10,
        'description': 'CPU模式 - 无GPU时使用'
    }
}

def get_optimization_recommendations() -> List[str]:
    """获取优化建议"""
    recommendations = [
        "1. 使用GPU加速 - 如果可用，GPU可以显著提升处理速度",
        "2. 增加批处理大小 - 更大的批次可以更好地利用GPU并行性",
        "3. 调整帧间距 - 跳过一些帧可以大幅提升速度",
        "4. 使用多线程 - 并行处理可以充分利用多核CPU",
        "5. 优化图像尺寸 - 较小的输入图像处理更快",
        "6. 启用混合精度 - FP16可以提升GPU性能并减少内存使用",
        "7. 使用TensorRT - 如果可用，TensorRT可以进一步优化推理速度",
        "8. 内存管理 - 定期清理GPU缓存避免内存不足"
    ]
    return recommendations

