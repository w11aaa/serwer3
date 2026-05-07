"""
优化后的视频处理模块
集成到现有Streamlit应用中
"""

import streamlit as App
import cv2 as CV2
import numpy as NP
import torch as Torch
from video_optimization import VideoOptimizer, MemoryOptimizer, PerformanceMonitor, OPTIMIZATION_CONFIGS
import time
import threading
from typing import List, Dict, Any

class OptimizedVideoProcessor:
    """优化的视频处理器"""
    
    def __init__(self):
        self.optimizer = None
        self.monitor = PerformanceMonitor()
        self.is_processing = False
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def process_video_optimized(self, video_path: str, model, config_name: str = 'balanced') -> List[Dict[str, Any]]:
        """使用优化配置处理视频"""
        if self.is_processing:
            raise Exception("视频处理正在进行中")
        
        self.is_processing = True
        
        try:
            # 获取优化配置
            config = OPTIMIZATION_CONFIGS.get(config_name, OPTIMIZATION_CONFIGS['balanced'])
            
            # 优化模型内存使用
            optimized_model = MemoryOptimizer.optimize_model_memory(model)
            
            # 创建优化器
            self.optimizer = VideoOptimizer(
                model=optimized_model,
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                use_gpu=config['use_gpu']
            )
            
            # 开始性能监控
            self.monitor.start_monitoring('total_processing')
            
            # 处理视频
            results = self.optimizer.optimized_video_processing(
                video_path=video_path,
                frame_spacing=config['frame_spacing']
            )
            
            # 结束监控
            self.monitor.end_monitoring('total_processing')
            
            return results
            
        finally:
            self.is_processing = False
            # 清理GPU缓存
            MemoryOptimizer.clear_gpu_cache()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.optimizer:
            return {
                **self.optimizer.stats,
                'performance_report': self.monitor.get_report()
            }
        return {}

def show_optimization_settings():
    """显示优化设置界面"""
    App.markdown("### ⚡ 性能优化设置")
    
    # 优化模式选择
    optimization_mode = App.selectbox(
        "选择优化模式",
        list(OPTIMIZATION_CONFIGS.keys()),
        format_func=lambda x: f"{x} - {OPTIMIZATION_CONFIGS[x]['description']}",
        help="不同的优化模式针对不同的使用场景"
    )
    
    # 显示当前配置
    config = OPTIMIZATION_CONFIGS[optimization_mode]
    App.markdown(f"""
    <div class="card">
        <h4>当前配置: {optimization_mode}</h4>
        <p><strong>批处理大小</strong>: {config['batch_size']}</p>
        <p><strong>工作线程数</strong>: {config['num_workers']}</p>
        <p><strong>GPU加速</strong>: {'是' if config['use_gpu'] else '否'}</p>
        <p><strong>帧间距</strong>: {config['frame_spacing']}</p>
        <p><strong>描述</strong>: {config['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return optimization_mode

def show_performance_recommendations():
    """显示性能优化建议"""
    from video_optimization import get_optimization_recommendations
    
    App.markdown("### 💡 性能优化建议")
    
    recommendations = get_optimization_recommendations()
    
    for i, rec in enumerate(recommendations, 1):
        App.markdown(f"**{rec}**")
    
    # 系统信息
    App.markdown("### 🖥️ 系统性能信息")
    
    col1, col2, col3 = App.columns(3)
    
    with col1:
        App.metric("CPU核心数", f"{App.session_state.get('cpu_count', 'N/A')}")
    
    with col2:
        cuda_available = Torch.cuda.is_available()
        App.metric("GPU可用", "是" if cuda_available else "否")
    
    with col3:
        if cuda_available:
            gpu_name = Torch.cuda.get_device_name(0)
            App.metric("GPU型号", gpu_name[:20] + "..." if len(gpu_name) > 20 else gpu_name)
        else:
            App.metric("GPU型号", "N/A")

def process_video_with_optimization(video_path: str, model, optimization_mode: str = 'balanced'):
    """使用优化设置处理视频"""
    processor = OptimizedVideoProcessor()
    
    # 创建进度条
    progress_bar = App.progress(0)
    status_text = App.empty()
    
    def update_progress(progress: float, status: str):
        progress_bar.progress(progress)
        status_text.text(status)
    
    processor.set_progress_callback(update_progress)
    
    try:
        # 开始处理
        status_text.text("🚀 开始优化视频处理...")
        
        # 处理视频
        results = processor.process_video_optimized(video_path, model, optimization_mode)
        
        # 获取性能统计
        stats = processor.get_performance_stats()
        
        # 显示结果
        status_text.text("✅ 视频处理完成！")
        progress_bar.progress(1.0)
        
        # 显示性能统计
        App.markdown("### 📊 处理性能统计")
        
        col1, col2, col3, col4 = App.columns(4)
        
        with col1:
            App.metric("处理帧数", stats.get('processed_frames', 0))
        
        with col2:
            App.metric("处理时间", f"{stats.get('processing_time', 0):.2f}s")
        
        with col3:
            App.metric("平均FPS", f"{stats.get('fps', 0):.2f}")
        
        with col4:
            total_frames = stats.get('total_frames', 0)
            processed_frames = stats.get('processed_frames', 0)
            if total_frames > 0:
                efficiency = (processed_frames / total_frames) * 100
                App.metric("处理效率", f"{efficiency:.1f}%")
            else:
                App.metric("处理效率", "N/A")
        
        return results
        
    except Exception as e:
        status_text.text(f"❌ 处理失败: {str(e)}")
        progress_bar.progress(0)
        raise e

# 集成到现有系统的辅助函数
def get_optimized_video_processing_function():
    """获取优化后的视频处理函数"""
    return process_video_with_optimization

def show_optimization_interface():
    """显示优化界面"""
    show_performance_recommendations()
    optimization_mode = show_optimization_settings()
    return optimization_mode

