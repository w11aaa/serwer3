"""
数据库集成模块
将数据库功能集成到Streamlit应用中
"""

import streamlit as App
from database import db_ops
from db_service import (
    detection_service, model_service, statistics_service, 
    user_service, audit_service, workorder_service, segment_service,
    config_service
)
from Export import GetWorkOrderPDF, GetWorkOrderExcel
from datetime import datetime
import json
import os as OS
import time

class DatabaseIntegration:
    """数据库集成类"""
    
    def __init__(self):
        self._initialized = False
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        if self._initialized:
            return
            
        try:
            db_ops.init_database()
            
            # 初始化默认系统配置
            self._init_default_configs()
            
            self._initialized = True
            # 只在第一次初始化时显示成功消息
            if not hasattr(App.session_state, 'db_init_message_shown'):
                App.success("✅ 数据库初始化成功")
                App.session_state['db_init_message_shown'] = True
        except Exception as e:
            App.error(f"❌ 数据库初始化失败: {str(e)}")
            # 记录错误到session state，避免重复显示
            App.session_state['db_init_error'] = str(e)

    def _init_default_configs(self):
        """初始化默认系统配置项"""
        configs = [
            ("risk_threshold_high", 0.08, "number", "高风险变形率阈值"),
            ("risk_threshold_mid", 0.03, "number", "中风险变形率阈值"),
            ("health_penalty_high", 1.5, "number", "高风险帧扣分系数(每1%)"),
            ("health_penalty_mid", 0.5, "number", "中风险帧扣分系数(每1%)"),
            ("health_limit_high", 60.0, "number", "高风险扣分上限"),
            ("health_limit_mid", 25.0, "number", "中风险扣分上限"),
            ("enable_auto_audit", False, "boolean", "是否启用自动验收审核"),
            ("system_name", "智慧管网AI分析平台", "string", "系统显示名称"),
        ]
        for key, val, cfg_type, desc in configs:
            # get_config 如果不存在会调用 set_config 创建
            config_service.get_config(key, default=val, config_type=cfg_type)
            # 确保描述信息存在
            config_service.set_config(key, val, config_type=cfg_type, description=desc)

    def save_image_analysis(self, image_path, analysis_result, plot_image, mask_image, segment_id=None):
        """保存图像分析结果"""
        try:
            # 创建任务
            from datetime import datetime
            task_name = f"图像分析_{OS.path.basename(image_path)}" if image_path else f"巡检报修_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task_id = detection_service.create_task(
                task_name=task_name,
                task_type="image",
                input_file_path=image_path or "N/A",
                user_id=1,  # 默认用户ID
                model_id=1,  # 默认模型ID
                segment_id=segment_id,
                parameters={"analysis_type": "single_image"}
            )
            
            # 更新任务状态
            detection_service.update_task_status(task_id, "processing")
            
            # 保存结果
            images_data = {
                "original": None,  # 原始图像路径已知，不存储
                "segmented": plot_image,
                "mask": mask_image
            }
            
            result_id = detection_service.save_detection_result(
                task_id=task_id,
                frame_number=1,
                analysis_data=analysis_result,
                images_data=images_data
            )
            
            # 完成任务
            detection_service.update_task_status(task_id, "completed", 100.0)
            
            return task_id, result_id
            
        except Exception as e:
            App.error(f"保存图像分析结果失败: {str(e)}")
            return None, None
    
    def save_video_analysis(self, video_path, video_results, processing_parameters, segment_id=None, existing_task_id=None):
        """保存视频分析结果"""
        try:
            # 创建任务（如果没提供）
            if existing_task_id:
                task_id = existing_task_id
            else:
                task_id = detection_service.create_task(
                    task_name=f"视频分析_{OS.path.basename(video_path)}",
                    task_type="video",
                    input_file_path=video_path,
                    user_id=1,  # 默认用户ID
                    model_id=1,  # 默认模型ID
                    segment_id=segment_id,
                    parameters=processing_parameters
                )
            
            # 更新任务状态
            detection_service.update_task_status(task_id, "processing")
            
            # 保存所有帧的结果
            for i, result_data in enumerate(video_results):
                images_data = {
                    "segmented": result_data.get("Plot"),
                    "mask": result_data.get("Mask"),
                    "output": result_data.get("Draw")
                }
                
                # 构建分析数据
                analysis_data = {
                    "Shape": [result_data.get("Shape")],
                    "AspectRatio": [result_data.get("AspectRatio")],
                    "Orientation": [result_data.get("Orientation")],
                    "Deformation": [result_data.get("Deformation")],
                    "Conclusion": ["Normal" if result_data.get("Shape") == "Circle" else "Deformed"],
                    "RiskLevel": [result_data.get("RiskLevel")],
                    "ActionSuggestion": [result_data.get("ActionSuggestion")]
                }
                
                detection_service.save_detection_result(
                    task_id=task_id,
                    frame_number=i + 1,
                    analysis_data=analysis_data,
                    images_data=images_data
                )
                
                # 更新进度
                progress = ((i + 1) / len(video_results)) * 100
                detection_service.update_task_status(task_id, "processing", progress)
            
            # 完成任务
            detection_service.update_task_status(task_id, "completed", 100.0)
            
            return task_id
            
        except Exception as e:
            App.error(f"保存视频分析结果失败: {str(e)}")
            return None
    
    def get_analysis_history(self, limit=10):
        """获取分析历史"""
        try:
            session = detection_service.db_manager.get_session()
            try:
                from database import DetectionTask, DetectionResult
                
                tasks = session.query(DetectionTask).order_by(
                    DetectionTask.created_at.desc()
                ).limit(limit).all()
                
                history = []
                for task in tasks:
                    # 获取任务结果统计
                    result_count = session.query(DetectionResult).filter(
                        DetectionResult.task_id == task.id
                    ).count()
                    
                    history.append({
                        'id': task.id,
                        'name': task.task_name,
                        'type': task.task_type,
                        'status': task.status,
                        'progress': task.progress,
                        'result_count': result_count,
                        'created_at': task.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(task.created_at, 'strftime') else str(task.created_at),
                        'completed_at': task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if task.completed_at and hasattr(task.completed_at, 'strftime') else (str(task.completed_at) if task.completed_at else None)
                    })
                
                return history
                
            finally:
                detection_service.db_manager.close_session(session)
                
        except Exception as e:
            App.error(f"获取分析历史失败: {str(e)}")
            return []
    
    def get_task_details(self, task_id):
        """获取任务详情"""
        try:
            session = detection_service.db_manager.get_session()
            try:
                from database import DetectionTask, DetectionResult
                
                task = session.query(DetectionTask).filter(DetectionTask.id == task_id).first()
                if not task:
                    return None
                
                results = session.query(DetectionResult).filter(
                    DetectionResult.task_id == task_id
                ).order_by(DetectionResult.frame_number).all()
                
                return {
                    'task': task,
                    'results': results
                }
                
            finally:
                detection_service.db_manager.close_session(session)
                
        except Exception as e:
            App.error(f"获取任务详情失败: {str(e)}")
            return None
    
    def get_statistics_dashboard(self):
        """获取统计仪表板数据"""
        try:
            # 获取任务统计
            task_stats = statistics_service.get_task_statistics(30)
            
            # 获取性能指标
            performance_metrics = statistics_service.get_performance_metrics(7)
            
            return {
                'task_statistics': task_stats,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            App.error(f"获取统计数据失败: {str(e)}")
            return None
    
    def export_task_data(self, task_id, export_format='json'):
        """导出任务数据"""
        try:
            if export_format == 'pdf':
                return self.export_task_pdf(task_id)
                
            task_details = self.get_task_details(task_id)
            if not task_details:
                return None
            
            task = task_details['task']
            results = task_details['results']
            
            if export_format == 'json':
                export_data = {
                    'task_info': {
                        'id': task.id,
                        'name': task.task_name,
                        'type': task.task_type,
                        'status': task.status,
                        'created_at': task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
                        'completed_at': task.completed_at.isoformat() if task.completed_at and hasattr(task.completed_at, 'isoformat') else (str(task.completed_at) if task.completed_at else None),
                        'parameters': json.loads(task.parameters) if task.parameters else None
                    },
                    'results': []
                }
                
            for result in results:
                extra_result_data = {}
                try:
                    extra_result_data = json.loads(result.result_data) if result.result_data else {}
                except Exception:
                    extra_result_data = {}

                risk_level = extra_result_data.get("RiskLevel")
                action_suggestion = extra_result_data.get("ActionSuggestion")
                if isinstance(risk_level, list):
                    risk_level = risk_level[0] if risk_level else None
                if isinstance(action_suggestion, list):
                    action_suggestion = action_suggestion[0] if action_suggestion else None

                result_data = {
                    'frame_number': result.frame_number,
                    'shape': result.shape,
                    'aspect_ratio': result.aspect_ratio,
                    'orientation': result.orientation,
                    'deformation': result.deformation,
                    'conclusion': result.conclusion,
                    'risk_level': risk_level,
                    'action_suggestion': action_suggestion,
                    'created_at': result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at)
                }
                export_data['results'].append(result_data)
                
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            
            return None
            
        except Exception as e:
            App.error(f"导出任务数据失败: {str(e)}")
            return None

    def export_task_pdf(self, task_id):
        """导出任务PDF报告"""
        try:
            from Export import GetPDF
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            task_details = self.get_task_details(task_id)
            if not task_details:
                return None
            
            task = task_details['task']
            results = task_details['results']
            
            # 构建报告元数据
            meta = {
                "project": task.task_name or "下水道检测报告",
                "operator": task.operator_name if hasattr(task, 'operator_name') and task.operator_name else "管理员",
                "date": task.created_at.strftime('%Y-%m-%d') if task.created_at else "—",
                "segment": "—", # 默认占位
                "report_type": task.task_type or "分析报告",
                "notes": "",
                "conclusion": "优良",
                "score": None
            }
            
            # 尝试从参数中获取更多信息
            if task.parameters:
                try:
                    params = json.loads(task.parameters)
                    if 'segment' in params: meta['segment'] = params['segment']
                    if 'operator' in params: meta['operator'] = params['operator']
                    if 'notes' in params: meta['notes'] = params['notes']
                except: pass

            data_for_pdf = []
            high_risk_count = 0
            
            for result in results:
                extra_result_data = {}
                try:
                    extra_result_data = json.loads(result.result_data) if result.result_data else {}
                except Exception:
                    extra_result_data = {}

                risk_level = extra_result_data.get("RiskLevel")
                if isinstance(risk_level, list):
                    risk_level = risk_level[0] if risk_level else None
                
                if risk_level == "高": high_risk_count += 1
                
                action_suggestion = extra_result_data.get("ActionSuggestion")
                if isinstance(action_suggestion, list):
                    action_suggestion = action_suggestion[0] if action_suggestion else None

                # Get images
                images = detection_service.get_result_images(result.id)
                plot_img = None
                mask_img = None
                
                for img in images:
                    if img.image_type == 'segmented': 
                        pil_img = Image.open(BytesIO(img.image_data))
                        plot_img = np.array(pil_img)
                    elif img.image_type == 'output': 
                        pil_img = Image.open(BytesIO(img.image_data))
                        mask_img = np.array(pil_img)
                    elif img.image_type == 'mask' and mask_img is None:
                        pil_img = Image.open(BytesIO(img.image_data))
                        mask_img = np.array(pil_img)

                # If missing images, create black ones
                if plot_img is None: plot_img = np.zeros((100, 100, 3), dtype=np.uint8)
                if mask_img is None: mask_img = np.zeros((100, 100, 3), dtype=np.uint8)

                item = {
                    "Plot": plot_img,
                    "Draw": mask_img,
                    "Shape": result.shape,
                    "AspectRatio": result.aspect_ratio,
                    "Orientation": result.orientation,
                    "Deformation": result.deformation,
                    "RiskLevel": risk_level,
                    "ActionSuggestion": action_suggestion,
                }
                data_for_pdf.append(item)
            
            if not data_for_pdf:
                return None
            
            meta["total"] = len(data_for_pdf)
            meta["high_risk"] = high_risk_count
            if high_risk_count > 0: meta["conclusion"] = "警告"

            return GetPDF(data_for_pdf, meta=meta)
            
        except Exception as e:
            App.error(f"导出PDF失败: {str(e)}")
            return None
    
    def export_batch_tasks(self, task_ids, export_format='PDF'):
        """批量导出任务报告"""
        try:
            from io import BytesIO
            import zipfile
            from Export import GetPDF
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for task_id in task_ids:
                    if export_format == 'PDF':
                        pdf_buffer = self.export_task_pdf(task_id)
                        if pdf_buffer:
                            task = detection_service.get_task(task_id)
                            filename = f"报告_{task.task_name or task_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
                            zip_file.writestr(filename, pdf_buffer.getvalue())
                    elif export_format == 'Excel':
                        # 假设有一个 export_task_excel 方法，如果没有则跳过
                        if hasattr(self, 'export_task_excel'):
                            excel_buffer = self.export_task_excel(task_id)
                            if excel_buffer:
                                task = detection_service.get_task(task_id)
                                filename = f"报告_{task.task_name or task_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
                                zip_file.writestr(filename, excel_buffer.getvalue())
            
            zip_buffer.seek(0)
            return zip_buffer
        except Exception as e:
            App.error(f"批量导出失败: {str(e)}")
            return None

    def cleanup_old_data(self, days=30):
        """清理指定天数前的历史数据"""
        try:
            session = detection_service.db_manager.get_session()
            try:
                from database import DetectionTask, DetectionResult, ResultImage
                from datetime import datetime, timedelta
                
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # 获取要删除的任务
                old_tasks = session.query(DetectionTask).filter(
                    DetectionTask.created_at < cutoff_date
                ).all()
                
                deleted_tasks = 0
                deleted_results = 0
                deleted_images = 0
                
                # 如果没有要删除的任务，直接返回
                if not old_tasks:
                    return {
                        'deleted_tasks': 0,
                        'deleted_results': 0,
                        'deleted_images': 0
                    }
                
                for task in old_tasks:
                    # 获取该任务的所有结果
                    results = session.query(DetectionResult).filter(
                        DetectionResult.task_id == task.id
                    ).all()
                    
                    # 删除每个结果的相关图像
                    for result in results:
                        result_images = session.query(ResultImage).filter(
                            ResultImage.result_id == result.id
                        ).all()
                        
                        for img in result_images:
                            session.delete(img)
                            deleted_images += 1
                        
                        # 删除结果
                        session.delete(result)
                        deleted_results += 1
                    
                    # 删除任务
                    session.delete(task)
                    deleted_tasks += 1
                
                # 提交所有更改
                session.commit()
                
                # 记录审计日志
                user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
                audit_service.log_action(user_info["id"], user_info["username"], "清理数据", "数据库管理", f"执行了历史数据清理（{days}天前）")
                
                return {
                    'deleted_tasks': deleted_tasks,
                    'deleted_results': deleted_results,
                    'deleted_images': deleted_images
                }
                
            except Exception as e:
                # 如果出错，回滚事务
                session.rollback()
                raise e
            finally:
                detection_service.db_manager.close_session(session)
                
        except Exception as e:
            App.error(f"数据清理失败: {str(e)}")
            return None

# 全局数据库集成实例
db_integration = DatabaseIntegration()

@App.fragment
def show_database_dashboard():
    """显示数据库仪表板"""
    App.markdown("### 📊 数据库仪表板")
    
    # 检查数据库初始化状态
    if hasattr(App.session_state, 'db_init_error'):
        App.error(f"数据库连接错误: {App.session_state['db_init_error']}")
        App.info("请检查数据库文件是否存在或重新启动应用")
        return
    
    # 获取统计数据
    try:
        dashboard_data = db_integration.get_statistics_dashboard()
        if not dashboard_data:
            App.warning("⚠️ 暂无统计数据，请先进行一些分析任务")
            return
        
        task_stats = dashboard_data['task_statistics']
        performance_metrics = dashboard_data['performance_metrics']
        
        # 显示统计卡片
        col1, col2, col3, col4 = App.columns(4)
        
        with col1:
            App.metric(
                "总任务数",
                task_stats.get('total_tasks', 0),
                delta=f"成功率: {task_stats.get('success_rate', 0):.1f}%"
            )
        
        with col2:
            App.metric(
                "完成任务",
                task_stats.get('completed_tasks', 0),
                delta=f"失败: {task_stats.get('failed_tasks', 0)}"
            )
        
        with col3:
            App.metric(
                "图像任务",
                task_stats.get('image_tasks', 0),
                delta=f"视频任务: {task_stats.get('video_tasks', 0)}"
            )
        
        with col4:
            avg_time = performance_metrics.get('average_processing_time', 0)
            App.metric(
                "检测结果",
                task_stats.get('total_results', 0),
                delta=f"平均处理时间: {avg_time:.1f}s"
            )
        
            
    except Exception as e:
        App.error(f"获取统计数据时发生错误: {str(e)}")
        App.info("请检查数据库连接或联系管理员")

@App.fragment
def show_analysis_history():
    """历史档案"""
    App.markdown("### 📂 历史档案查询")
    
    user_info = App.session_state.get("user_info", {})
    
    # 检查数据库连接
    if hasattr(App.session_state, 'db_init_error'):
        App.error("数据库连接错误，无法显示历史记录")
        return
    
    try:
        # 获取历史记录
        history = db_integration.get_analysis_history(50)
        if not history:
            App.info("📝 暂无分析历史记录，请先进行一些分析任务")
            return
        
        # 批量操作栏
        col_batch1, col_batch2, col_batch3 = App.columns([3, 1, 1])
        with col_batch1:
            all_ids = [t['id'] for t in history]
            
            # 使用回调函数同步全选状态
            def on_all_history_change():
                if App.session_state.get("history_select_all"):
                    App.session_state["history_multiselect"] = all_ids
                else:
                    App.session_state["history_multiselect"] = []

            is_all_history = App.checkbox("全选所有记录", key="history_select_all", on_change=on_all_history_change)
            
            selected_rows = App.multiselect(
                "批量选择任务 (支持导出与删除)",
                options=all_ids,
                format_func=lambda x: f"ID:{x} - {next((t['name'] for t in history if t['id'] == x), '未知')}",
                key="history_multiselect"
            )
        with col_batch2:
            if selected_rows:
                export_fmt = App.selectbox("导出格式", ["PDF", "Excel"], label_visibility="collapsed")
                if App.button(f"📦 批量导出 {len(selected_rows)} 项", type="primary", use_container_width=True):
                    with App.spinner("正在打包报告..."):
                        zip_buf = db_integration.export_batch_tasks(selected_rows, export_format=export_fmt)
                        if zip_buf:
                            App.download_button(
                                "💾 点击下载 ZIP",
                                zip_buf,
                                f"批量报告_{datetime.now().strftime('%Y%m%d')}.zip",
                                "application/zip",
                                use_container_width=True
                            )
        with col_batch3:
            if selected_rows:
                App.markdown("&nbsp;", unsafe_allow_html=True)
                confirm_del = App.checkbox("确认删除任务", key="history_del_check")
                if confirm_del:
                    if App.button("❗ 确认彻底删除", type="primary", use_container_width=True):
                        deleted = detection_service.delete_tasks(selected_rows)
                        audit_service.log_action(user_info.get("id"), user_info.get("username", "Unknown"), "批量删除", "历史档案", 
                                               f"批量删除了 {deleted} 条检测任务")
                        App.success(f"已成功删除 {deleted} 项历史记录")
                        if "history_del_check" in App.session_state:
                            del App.session_state["history_del_check"]
                        time.sleep(1)
                        App.rerun()

        # 显示历史表格
        App.dataframe(
            history,
            width="stretch",
            hide_index=True,
            column_config={
                "id": App.column_config.NumberColumn("任务ID"),
                "name": App.column_config.TextColumn("任务名称"),
                "type": App.column_config.TextColumn("类型"),
                "status": App.column_config.TextColumn("状态"),
                "progress": App.column_config.ProgressColumn("进度"),
                "result_count": App.column_config.NumberColumn("结果数"),
                "created_at": App.column_config.DatetimeColumn("创建时间"),
                "completed_at": App.column_config.DatetimeColumn("完成时间")
            }
        )
        
        # 任务详情查看
        if history:
            # 处理跳转逻辑
            default_index = 0
            jump_id = App.session_state.get("history_selected_task_id")
            task_ids = [task['id'] for task in history]
            if jump_id and jump_id in task_ids:
                default_index = task_ids.index(jump_id)
                # 只有第一次进入时展开，或者持续展开
                # App.session_state.pop("history_selected_task_id", None) # 如果希望只生效一次

            selected_task_id = App.selectbox(
                "选择任务查看详情",
                task_ids,
                index=default_index,
                format_func=lambda x: f"任务 {x} - {next((t['name'] for t in history if t['id'] == x), '未知')}"
            )
            
            if selected_task_id:
                try:
                    task_details = db_integration.get_task_details(selected_task_id)
                    if task_details:
                        # 如果是跳转过来的，默认展开详情
                        is_jump = (jump_id == selected_task_id)
                        with App.expander("📋 任务详情", expanded=is_jump):
                            task = task_details['task']
                            results = task_details['results']
                            
                            # 基本信息
                            col1, col2 = App.columns(2)
                            with col1:
                                App.write(f"**任务名称**: {task.task_name}")
                                App.write(f"**任务类型**: {task.task_type}")
                                App.write(f"**状态**: {task.status}")
                            with col2:
                                App.write(f"**进度**: {task.progress}%")
                                App.write(f"**结果数量**: {len(results)}")
                                if task.completed_at:
                                    App.write(f"**完成时间**: {task.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # 参数信息
                            if task.parameters:
                                try:
                                    params = json.loads(task.parameters)
                                    App.write("**分析参数**:")
                                    App.json(params)
                                except:
                                    App.write(f"**分析参数**: {task.parameters}")
                            
                            # 导出功能
                            export_col1, export_col2, export_col3 = App.columns(3)
                            with export_col1:
                                if App.button("📥 导出原始数据", key=f"export_json_{selected_task_id}"):
                                    export_data = db_integration.export_task_data(selected_task_id)
                                    if export_data:
                                        App.download_button(
                                            "下载原始数据",
                                            export_data,
                                            f"task_{selected_task_id}_data.json",
                                            "application/json"
                                        )
                                    else:
                                        App.error("导出失败，请重试")
                            
                            with export_col2:
                                if App.button("📄 导出PDF报告", key=f"export_pdf_{selected_task_id}"):
                                    with App.spinner("正在生成PDF报告..."):
                                        pdf_buffer = db_integration.export_task_data(selected_task_id, export_format='pdf')
                                        if pdf_buffer:
                                            App.download_button(
                                                "下载PDF报告",
                                                pdf_buffer,
                                                f"task_{selected_task_id}_report.pdf",
                                                "application/pdf"
                                            )
                                        else:
                                            App.error("PDF生成失败或无数据")

                            with export_col3:
                                if App.button("🗑️ 删除任务", key=f"delete_{selected_task_id}"):
                                    App.session_state[f"confirm_del_{selected_task_id}"] = True
                                
                                if App.session_state.get(f"confirm_del_{selected_task_id}"):
                                    App.warning("⚠️ 此操作将永久删除该任务及其所有相关数据")
                                    if App.button("❗ 确认彻底删除", key=f"confirm_btn_{selected_task_id}", type="primary"):
                                        if detection_service.delete_tasks([selected_task_id]):
                                            audit_service.log_action(user_info.get("id"), user_info.get("username", "Unknown"), 
                                                                   "删除任务", "历史档案", f"删除了任务 {selected_task_id}")
                                            App.success("任务已删除")
                                            del App.session_state[f"confirm_del_{selected_task_id}"]
                                            time.sleep(0.5)
                                            App.rerun()
                                        else:
                                            App.error("删除失败")
                    
                    else:
                        App.warning("⚠️ 无法获取任务详情")
                        
                except Exception as e:
                    App.error(f"获取任务详情时发生错误: {str(e)}")
                    
    except Exception as e:
        App.error(f"获取历史记录时发生错误: {str(e)}")
        App.info("请检查数据库连接或联系管理员")


def show_workorder_dashboard(assignee_id=None):
    """显示任务管理模块"""
    App.markdown("### 📋 任务处理中心")

    if hasattr(App.session_state, 'db_init_error'):
        App.error("数据库连接错误，无法加载任务模块")
        return

    try:
        user_info = App.session_state.get("user_info", {})
        # 获取基础统计
        stats = workorder_service.get_work_order_stats(assignee_id)
        status_distribution = stats.get("status_distribution", {})
        
        # 检查是否有待紧急核准的任务 (仅统计待审核状态且标记为强制核准的任务)
        session = workorder_service.db_manager.get_session()
        try:
            from database import WorkOrder as _WO
            total_forced = session.query(_WO).filter(_WO.is_forced == True, _WO.status == "待审核").count()
        finally:
            workorder_service.db_manager.close_session(session)

        # 管理员端显示统计指标 (折叠显示以节省空间)
        if not assignee_id:
            with App.expander("📊 任务概览与统计", expanded=False):
                c1, c2, c3, c4, c5 = App.columns(5)
                c1.metric("总任务", stats.get("total", 0))
                c2.metric("已完结", stats.get("closed", 0), f"完结率 {stats.get('closed_rate', 0):.1f}%")
                
                # 待指派
                c3.metric("待派单", status_distribution.get("待指派", 0))
                
                # 处理中
                c4.metric("处理中", status_distribution.get("进行中", 0))
                
                # 待验收 (增加联动展示)
                audit_count = status_distribution.get("待审核", 0)
                if total_forced > 0:
                    c5.metric("待验收", audit_count, f"🚨 {total_forced} 项紧急核准", delta_color="inverse")
                else:
                    c5.metric("待验收", audit_count)

        if not assignee_id:
            _workorder_dispatch_command_center()
        else:
            _workorder_list_and_update(assignee_id=assignee_id)

    except Exception as e:
        App.error(f"加载任务模块失败: {str(e)}")



def _workorder_dispatch_command_center():
    user_info = App.session_state.get("user_info", {})
    with App.expander("🛰️ 任务分派指挥中心 (全平铺高效控制台)", expanded=True):
        # 预先获取工人列表
        all_workers = user_service.list_users(role="worker", status="active")
        worker_options = [w.username for w in all_workers]
        worker_display = {w.username: (f"{w.full_name} ({w.username})" if w.full_name else w.username) for w in all_workers}
        
        # 1. 顶部：状态概览卡片 (平铺)
        stats = workorder_service.get_work_order_stats()
        status_dist = stats.get("status_distribution", {})
        
        c1, c2, c3, c4 = App.columns(4)
        with c1:
            if App.button(f"🔴 待指派: {status_dist.get('待指派', 0)}", key="cc_filter_pending", use_container_width=True):
                App.session_state["workorder_status_filter"] = "待指派"
                App.session_state["workorder_forced_filter"] = "全部"
                App.session_state["active_step_guide"] = "💡 已为您筛选【待指派】任务，请在下方台账中勾选并指派。"
                App.rerun()
        with c2:
            if App.button(f"🟡 处理中: {status_dist.get('进行中', 0)}", key="cc_filter_processing", use_container_width=True):
                App.session_state["workorder_status_filter"] = "处理中"
                App.session_state["workorder_forced_filter"] = "全部"
                App.session_state["active_step_guide"] = "💡 已为您筛选【处理中】任务，您可以查看进度或修改负责人。"
                App.rerun()
        with c3:
            # 检查是否有强制核准件
            session = workorder_service.db_manager.get_session()
            try:
                from database import WorkOrder as _WO
                forced_count = session.query(_WO).filter(_WO.is_forced == True, _WO.status == "待审核").count()
            finally:
                workorder_service.db_manager.close_session(session)
            
            btn_label = f"🔵 待验收: {status_dist.get('待审核', 0)}"
            if forced_count > 0:
                btn_label = f"🚨 待验收: {status_dist.get('待审核', 0)} ({forced_count}项核准)"
            
            if App.button(btn_label, key="cc_filter_audit", use_container_width=True):
                App.session_state["workorder_status_filter"] = "待验收"
                if forced_count > 0:
                    App.session_state["workorder_forced_filter"] = "强制核准/紧急任务"
                    App.session_state["active_step_guide"] = f"🚨 优先处理 {forced_count} 项【强制核准】任务，这些任务 AI 评分较低，需人工确认现场修复质量。"
                else:
                    App.session_state["workorder_forced_filter"] = "常规任务"
                    App.session_state["active_step_guide"] = "💡 已为您筛选【待验收】任务，请在下方详情中核对 AI 评分并结项。"
                App.rerun()
        with c4:
            if App.button(f"✅ 已完结: {status_dist.get('已完成', 0)}", key="cc_filter_completed", use_container_width=True):
                App.session_state["workorder_status_filter"] = "已完结"
                App.session_state["workorder_forced_filter"] = "全部"
                App.session_state["active_step_guide"] = "💡 已为您筛选【已完结】任务，您可以导出报表或查看历史归档。"
                App.rerun()
        
        # 增加操作指引提示
        if "active_step_guide" in App.session_state:
            App.info(App.session_state["active_step_guide"])
        
        App.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # ── 核心变动：左侧台账 + 右侧操作面板 ────────────────
        main_col_left, main_col_right = App.columns([2.2, 1])
        
        with main_col_left:
            App.markdown("#### 📋 任务台账 (实时联动)")
            selected_ids, selected_rows, order_rows = _render_workorder_ledger()
            
            # 切换选择时重置控制台状态
            if "prev_selected_ids" not in App.session_state:
                App.session_state["prev_selected_ids"] = []
            
            if App.session_state["prev_selected_ids"] != selected_ids:
                App.session_state["prev_selected_ids"] = selected_ids
                # 清理旧的控制台状态
                for k in ["unified_status_select", "unified_worker_select", "unified_notes_input"]:
                    if k in App.session_state:
                        del App.session_state[k]
        
        with main_col_right:
            App.markdown("#### ⚡ 操作面板")
            is_update_mode = len(selected_ids) > 0
            
            # 容器化操作区
            with App.container(border=True):
                if not is_update_mode:
                    # --- 创建模式 (方案 2: 侧边操作栏) ---
                    App.caption("🆕 下达新任务")
                    
                    # 1. 管段选择
                    segments = segment_service.list_segments()
                    seg_options = {s.id: f"{s.segment_code} — {s.location or '无位置'}" for s in segments}
                    if not seg_options:
                        App.error("❌ 无管段数据")
                        selected_seg_id = None
                    else:
                        selected_seg_id = App.selectbox("📍 施工管段 (必选)", list(seg_options.keys()), 
                                                      format_func=lambda x: seg_options[x], 
                                                      key="unified_seg_select")
                    
                    # 2. 负责人预指派
                    target_assignee = App.selectbox("👷 预指派负责人", ["(暂不指派)"] + worker_options, 
                                                  format_func=lambda x: worker_display.get(x, x), 
                                                  key="unified_worker_select")
                    assignee_val = None if target_assignee == "(暂不指派)" else target_assignee
                    
                    # 3. 备注
                    notes = App.text_input("📝 任务备注", placeholder="输入任务要求...", key="unified_notes_input")
                    
                    # 4. 巡检记录批量导入
                    history = db_integration.get_analysis_history(30)
                    task_options = [item["id"] for item in history if item.get("result_count", 0) > 0]
                    
                    if task_options:
                        selected_tasks = App.multiselect("🔍 导入巡检记录缺陷", task_options, 
                                                       format_func=lambda x: f"巡检{x}-{next((t['name'] for t in history if t['id'] == x), '未知')}", 
                                                       key="unified_tasks_select")
                        
                        btn_label = f"🚀 确认下达 ({len(selected_tasks)})" if selected_tasks else "🚀 确认下达任务"
                        btn_disabled = not selected_tasks or not selected_seg_id
                        if App.button(btn_label, type="primary", use_container_width=True, disabled=btn_disabled, key="unified_create_btn"):
                            total_created = 0
                            for task_id in selected_tasks:
                                count = workorder_service.create_work_orders_from_task(task_id, risk_level="高", segment_id=selected_seg_id, assignee=assignee_val)
                                total_created += count
                            if total_created > 0:
                                audit_service.log_action(user_info.get("id"), user_info.get("username"), "创建任务", "任务管理", f"从巡检记录批量创建了 {total_created} 个任务")
                                App.toast(f"✅ 已成功下达 {total_created} 个任务")
                                segment_service.update_last_inspected(selected_seg_id)
                                App.rerun()
                    else:
                        App.caption("💡 暂无可用巡检记录。")

                elif len(selected_ids) == 1:
                    # --- 单选操作模式 (方案 1: 动态按钮 & 方案 2: AI 对照集成) ---
                    task = selected_rows[0]
                    raw_cur_status = task.get("状态", "")
                    clean_status = ""
                    for opt in ["待指派", "处理中", "待验收", "已完结"]:
                        if opt in raw_cur_status:
                            clean_status = opt
                            break
                    
                    # 1. 顶部：AI 维修对比 (直接展示，不使用 expander 以实现 Master-Detail 效果)
                    _session = None
                    try:
                        _session = workorder_service.db_manager.get_session()
                        from database import WorkOrder as _WO, DetectionResult as _DR, ResultImage as _RI, DetectionTask as _DT
                        _wo_obj = _session.query(_WO).filter(_WO.id == selected_ids[0]).first()
                        
                        if _wo_obj:
                            App.markdown(f"#### 🔍 任务详情: `{task.get('任务编号')}`")
                            
                            # 确定“报修前”和“维修后”结果
                            pre_res = None
                            if hasattr(_wo_obj, 'initial_result_id') and _wo_obj.initial_result_id:
                                pre_res = _session.query(_DR).filter(_DR.id == _wo_obj.initial_result_id).first()
                            
                            if not pre_res and _wo_obj.segment_id:
                                pre_res = _session.query(_DR).join(_DT).filter(
                                    _DT.segment_id == _wo_obj.segment_id,
                                    _DR.created_at <= _wo_obj.created_at
                                ).order_by(_DR.created_at.desc()).first()

                            post_res = _wo_obj.result # 维修后结果
                            # 只有进入“待验收”或“已完结”状态，才展示维修后对比图
                            if clean_status in ["待指派", "处理中"]:
                                post_res = None
                            
                            # 兜底：如果报修前后的结果 ID 相同，说明尚未上传真正的维修后结果
                            if pre_res and post_res and pre_res.id == post_res.id:
                                post_res = None
                            
                            # 双栏对比或单栏展示
                            if pre_res or post_res:
                                img_c1, img_c2 = App.columns(2)
                                with img_c1:
                                    if pre_res:
                                        img_pre = _session.query(_RI).filter(_RI.result_id == pre_res.id).first()
                                        if img_pre: App.image(img_pre.image_data, caption="📸 报修前", use_container_width=True)
                                    else:
                                        App.caption("无报修前图像")
                                with img_c2:
                                    if post_res:
                                        img_post = _session.query(_RI).filter(_RI.result_id == post_res.id).first()
                                        if img_post: App.image(img_post.image_data, caption="📸 维修后", use_container_width=True)
                                    else:
                                        App.caption("尚未上传维修后图像")
                            
                            # 评分条
                            if post_res:
                                import re
                                saved_score_match = re.search(r'AI\s*得分[:：]\s*(\d+)', _wo_obj.processing_notes or "")
                                from HealthScore import calc_health_score
                                mock_res = [{"Deformation": post_res.deformation or 0, "RiskLevel": "中"}]
                                hs = calc_health_score(mock_res)
                                final_score = int(saved_score_match.group(1)) if saved_score_match else hs['score']
                                
                                App.progress(final_score / 100, text=f"AI 修复评分: {final_score} ({hs['level']})")
                                if _wo_obj.processing_notes:
                                    App.info(f"👷 施工说明: {_wo_obj.processing_notes}")

                    except Exception as e:
                        App.error(f"加载 AI 数据失败: {e}")
                    finally:
                        if _session: _session.close()

                    App.markdown("---")
                    
                    # 2. 中部：动态操作卡片
                    App.markdown(f"**当前状态:** `{clean_status or raw_cur_status}`")
                    
                    notes = App.text_input("📝 操作备注/意见", placeholder="输入处理要求或意见...", key="unified_notes_input")
                    
                    if clean_status == "待指派":
                        target_assignee = App.selectbox("👷 选择负责人", worker_options, 
                                                      format_func=lambda x: worker_display.get(x, x), 
                                                      key="unified_worker_select")
                        if App.button("👤 立即派单", type="primary", use_container_width=True):
                            workorder_service.update_work_order(selected_ids[0], status="进行中", assignee=target_assignee, processing_notes=notes or "管理员派单")
                            App.toast("✅ 已指派并开始处理")
                            App.rerun()
                            
                    elif clean_status == "处理中":
                        target_assignee = App.selectbox("👷 调整负责人", ["(保持现状)"] + worker_options, 
                                                      format_func=lambda x: worker_display.get(x, x), 
                                                      key="unified_worker_select")
                        assignee_val = None if target_assignee == "(保持现状)" else target_assignee
                        
                        c1, c2 = App.columns(2)
                        with c1:
                            if App.button("📤 进入验收", type="primary", use_container_width=True):
                                workorder_service.update_work_order(selected_ids[0], status="待审核", processing_notes=notes or "管理员转验收")
                                App.toast("✅ 已转为待验收状态")
                                App.rerun()
                        with c2:
                            if App.button("💾 仅存备注", use_container_width=True):
                                update_params = {}
                                if assignee_val: update_params["assignee"] = assignee_val
                                workorder_service.update_work_order(selected_ids[0], processing_notes=notes, **update_params)
                                App.toast("✅ 已更新")
                                App.rerun()

                    elif clean_status == "待验收":
                        c1, c2 = App.columns(2)
                        with c1:
                            if App.button("✅ 准予结项", type="primary", use_container_width=True):
                                workorder_service.update_work_order(selected_ids[0], status="已完成", processing_notes=notes or "通过验收")
                                App.toast("✅ 任务已结项")
                                App.rerun()
                        with c2:
                            if App.button("❌ 驳回重修", type="secondary", use_container_width=True):
                                workorder_service.update_work_order(selected_ids[0], status="进行中", processing_notes=f"【驳回】{notes or '请核对维修质量'}")
                                App.toast("⚠️ 任务已驳回")
                                App.rerun()
                    
                    elif clean_status == "已完结":
                        if App.button("🔄 重启任务", type="secondary", use_container_width=True):
                            workorder_service.update_work_order(selected_ids[0], status="待指派", processing_notes=notes or "管理员重启任务")
                            App.toast("✅ 任务已重启")
                            App.rerun()
                    
                    # 3. 底部：最近流转日志
                    logs = workorder_service.list_work_order_logs(selected_ids[0], limit=3)
                    if logs:
                        with App.expander("🕒 最近日志", expanded=False):
                            for log in logs:
                                App.caption(f"{log.created_at.strftime('%m-%d')} | {log.action_type}: {log.notes or '-'}")

                else:
                    # --- 批量操作模式 ---
                    App.markdown(f"**已选中:** `{len(selected_ids)}` 个任务")
                    target_assignee = App.selectbox("👷 批量指派负责人", ["(保持现状)"] + worker_options, 
                                                  format_func=lambda x: worker_display.get(x, x), 
                                                  key="unified_worker_select")
                    assignee_val = None if target_assignee == "(保持现状)" else target_assignee
                    
                    status_options = ["(保持现状)", "待指派", "处理中", "待验收", "已完结"]
                    next_status = App.selectbox("📈 批量变更状态", status_options, key="unified_status_select")
                    notes = App.text_input("📝 批量备注", placeholder="输入批量处理意见...", key="unified_notes_input")
                    
                    if App.button(f"⚡ 批量执行 ({len(selected_ids)})", type="primary", use_container_width=True):
                        update_params = {}
                        status_db_map = {"待指派": "待指派", "处理中": "进行中", "待验收": "待审核", "已完结": "已完成"}
                        if next_status != "(保持现状)": 
                            update_params["status"] = status_db_map.get(next_status, next_status)
                        if assignee_val: update_params["assignee"] = assignee_val
                        
                        workorder_service.batch_update_work_orders(selected_ids, notes=notes or "批量修改", **update_params)
                        App.toast(f"✅ 已成功批量处理 {len(selected_ids)} 个任务")
                        App.rerun()

            # --- 导出工具 (侧边底部) ---
            if is_update_mode:
                with App.expander("📄 数据导出"):
                    if App.button("📥 生成 Excel 报表", key="wo_side_gen_excel", use_container_width=True):
                        App.session_state["wo_export_xlsx"] = GetWorkOrderExcel(selected_rows).getvalue()
                    if App.session_state.get("wo_export_xlsx"):
                        App.download_button("⬇ 点击下载", data=App.session_state["wo_export_xlsx"],
                                          file_name=f"tasks_export_{len(selected_ids)}.xlsx",
                                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                          use_container_width=True)

    return order_rows


def _render_workorder_ledger(assignee_id=None):
    """
    渲染任务台账表格及筛选器，返回当前选中的任务 ID 列表和行数据。
    """
    user_info = App.session_state.get("user_info", {})
    
    with App.expander("🔍 筛选任务 (任务查询指引)"):
        col1, col2 = App.columns(2)
        with col1:
            status_filter = App.selectbox("任务状态", ["全部", "待指派", "处理中", "待验收", "已完结"], key="workorder_status_filter")
            # 内部状态映射回数据库原始值
            status_map = {"全部": "全部", "待指派": "待指派", "处理中": "进行中", "待验收": "待审核", "已完结": "已完成"}
            db_status = status_map.get(status_filter, "全部")

        with col2:
            forced_filter = App.selectbox("筛选类别", ["全部", "常规任务", "强制核准/紧急任务"], key="workorder_forced_filter")
    
    # 获取任务列表
    is_forced_val = None
    if forced_filter == "常规任务":
        is_forced_val = False
    elif forced_filter == "强制核准/紧急任务":
        is_forced_val = True
        
    orders = workorder_service.list_work_orders(status=db_status, is_forced=is_forced_val, limit=300)
    
    if assignee_id:
        # 如果是工人端，只显示分配给该工人的任务
        orders = [o for o in orders if o.assignee == App.session_state["user_info"]["username"]]
    
    if not orders:
        App.info("暂无任务记录")
        return [], [], []

    # 在列表上方标注强制核准的任务
    forced_count = len([o for o in orders if hasattr(o, 'is_forced') and o.is_forced])
    if forced_count > 0:
        App.markdown(f"🚨 **发现 <span style='color:#e74c3c'>{forced_count}</span> 项强制核准件 (AI 评分较低但人工确认通过)**", unsafe_allow_html=True)

    order_rows = []
    status_display_map = {
        "待指派": "待指派",
        "进行中": "处理中",
        "待审核": "待验收",
        "已完成": "已完结"
    }

    for order in orders:
        disp_status = status_display_map.get(order.status, order.status)
        status_text = f"[{disp_status}]"
        submission_type = "常规"
        if hasattr(order, 'is_forced') and order.is_forced:
            import re
            score_match = re.search(r'AI\s*得分[:：]\s*(\d+)', order.processing_notes or "")
            score = int(score_match.group(1)) if score_match else 100
            
            if score < 60:
                status_text = f"⚠️ {status_text} (评分预警)"
                submission_type = "🔴 低评分核准"
            else:
                status_text = f"🚨 {status_text} (紧急)"
                submission_type = "🚨 紧急核准"
            
        order_rows.append({
            "选择": False,
            "id": order.id,
            "类型": submission_type,
            "任务编号": order.order_code,
            "位置": order.segment.segment_code if order.segment else "",
            "segment_id": order.segment_id,
            "任务描述": order.title,
            "风险": order.risk_level,
            "优先级": order.priority,
            "状态": status_text,
            "谁在修": order.assignee or "未派单",
            "最新进展": order.processing_notes or "",
            "下发时间": order.created_at.strftime('%m-%d %H:%M') if order.created_at else "",
        })

    if assignee_id:
        App.info(f"您好，以下是为您指派的任务清单，请及时处理。")
    
    # ── 增加全选功能 ──
    is_all_selected = App.checkbox("全选当前筛选的任务", key=f"select_all_orders_{assignee_id or 'admin'}")
    if is_all_selected:
        for row in order_rows:
            row["选择"] = True
            
    edited = App.data_editor(
        order_rows,
        column_config={
            "选择": App.column_config.CheckboxColumn("选", default=False, width="small"),
            "id": None, 
            "类型": App.column_config.TextColumn("类型", width="small", disabled=True),
            "任务编号": App.column_config.TextColumn("编号", width="small", disabled=True),
            "位置": App.column_config.TextColumn("位置", width="small", disabled=True),
            "任务描述": App.column_config.TextColumn("任务描述", width="medium", disabled=True),
            "风险": App.column_config.SelectboxColumn("风险等级", options=["高", "中", "低"], width="small", disabled=True),
            "优先级": App.column_config.SelectboxColumn("优先级", options=["紧急", "高", "普通", "低"], width="small", disabled=True),
            "状态": App.column_config.TextColumn("状态", width="small", disabled=True),
            "谁在修": App.column_config.TextColumn("负责人", width="small", disabled=True),
            "最新进展": App.column_config.TextColumn("处理进展", width="medium", disabled=True),
            "下发时间": App.column_config.TextColumn("下发时间", width="small", disabled=True),
            "segment_id": None,
        },
        hide_index=True,
        use_container_width=True,
        key="workorder_table_editor",
    )

    selected_ids = [row["id"] for row in edited if row.get("选择")]
    selected_rows = [row for row in edited if row.get("选择")]
    
    # 同步到 session_state
    App.session_state["selected_workorder_ids"] = selected_ids
    App.session_state["selected_workorder_rows"] = selected_rows

    if selected_ids:
        # 清除操作指引
        if "active_step_guide" in App.session_state:
            del App.session_state["active_step_guide"]
            
    return selected_ids, selected_rows, order_rows


def _workorder_list_and_update(assignee_id=None, order_rows=None):
    user_info = App.session_state.get("user_info", {})
    if order_rows is None:
        App.markdown("#### 📋 任务台账")
        selected_ids, selected_rows, order_rows = _render_workorder_ledger(assignee_id=assignee_id)
    else:
        # 已经从指挥中心获取了 order_rows 和 session_state 中的选择结果
        selected_ids = App.session_state.get("selected_workorder_ids", [])
        selected_rows = App.session_state.get("selected_workorder_rows", [])
    
    if not order_rows:
        return

    if selected_ids:
        App.markdown(f"**已选 {len(selected_ids)} 个任务**")
        
        if user_info.get("role") == "admin":
            # 驳回对话框
            if App.session_state.get("show_reject_dialog"):
                with App.form("reject_reason_form"):
                    App.markdown("### ❌ 批量驳回重修")
                    quick_reasons = ["现场图像不清晰，无法核实结果", "关键部位修复质量未达标", "作业现场清理不彻底", "修复位置偏移，请核对"]
                    reason_select = App.selectbox("常用处理意见 (快速选择)", ["自定义输入"] + quick_reasons)
                    
                    reason_text = App.text_area("详细处理建议", 
                        value="" if reason_select == "自定义输入" else reason_select,
                        placeholder="请输入详细的驳回理由或修改建议")
                    
                    if App.form_submit_button("确认驳回"):
                        final_reason = reason_text if reason_text else reason_select
                        target_ids = App.session_state["show_reject_dialog"]
                        count = workorder_service.batch_update_work_orders(
                            target_ids, 
                            status="进行中", 
                            notes=f"【管理员驳回】建议：{final_reason}"
                        )
                        if count:
                            audit_service.log_action(user_info["id"], user_info["username"], "批量驳回", "任务管理", 
                                                   f"驳回了 {count} 个任务。建议：{final_reason}")
                            App.toast(f"✅ 已驳回 {count} 条任务")
                            del App.session_state["show_reject_dialog"]
                            App.rerun()
                    if App.button("取消"):
                        del App.session_state["show_reject_dialog"]
                        App.rerun()

        export_col, _ = App.columns([2, 3])
        with export_col:
            with App.expander("📄 导出 Excel"):
                if App.button("📥 导出 Excel", key="wo_btn_gen_excel", use_container_width=True):
                    App.session_state["wo_export_xlsx"] = GetWorkOrderExcel(selected_rows).getvalue()
                if App.session_state.get("wo_export_xlsx"):
                    App.download_button(
                        "⬇ 下载表格",
                        data=App.session_state["wo_export_xlsx"],
                        file_name="tasks.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="wo_btn_dl_excel",
                        use_container_width=True,
                    )

    # 用原始 order_rows（未编辑的 id 列表）构建任务流转
    orig_ids = [row["id"] for row in order_rows]

    # 联动：如果用户在表格中勾选了行，自动切换下方详情到选中的第一行
    default_index = 0
    if selected_ids:
        try:
            default_index = orig_ids.index(selected_ids[0])
        except ValueError:
            default_index = 0

    App.markdown("#### 🔄 任务详情")
    if selected_ids:
        App.caption(f"💡 已根据勾选自动定位到任务: {selected_ids[0]}")

    selected_order_id = App.selectbox("选择任务", orig_ids, 
        index=default_index,
        key="workorder_selected_id",
        format_func=lambda x: next((f"{r['任务编号']} - {r['任务描述']}" for r in order_rows if r["id"] == x), str(x)))

    # 切换任务时重置输入框的 session_state，避免旧值残留
    prev_key = "workorder_prev_selected_id"
    if App.session_state.get(prev_key) != selected_order_id:
        App.session_state[prev_key] = selected_order_id
        for k in ["workorder_assignee", "workorder_notes", "workorder_next_status"]:
            if k in App.session_state:
                del App.session_state[k]

    current_order = next((item for item in order_rows if item["id"] == selected_order_id), None)
    if current_order:
        cur_s = current_order["状态"]
        # 预先获取数据库对象，供后续 UI 使用
        _wo = None
        try:
            _session = workorder_service.db_manager.get_session()
            try:
                from database import WorkOrder as _WO, DetectionResult as _DR, ResultImage as _RI, DetectionTask as _DT
                _wo = _session.query(_WO).filter(_WO.id == selected_order_id).first()
                
                # 初始化备注缓存（仅在未被用户编辑时）
                if "workorder_notes" not in App.session_state:
                    App.session_state["workorder_notes"] = _wo.processing_notes or "" if _wo else ""
                
                if "workorder_assignee" not in App.session_state:
                    App.session_state["workorder_assignee"] = current_order.get("负责人") or ""

                # ── 维修/对比状态展示 ──────────────────────────
                # 只要有结果 ID (不管是初始的还是最终的) 就显示分析区域
                if _wo and (_wo.result_id or (getattr(_wo, 'initial_result_id', None))):
                    App.markdown("#### 🔍 AI 分析与维修对比")
                    
                    # 1. 确定“报修前”和“维修后”两个结果
                    pre_result = None
                    post_result = None
                    
                    # 维修后结果始终是当前任务关联的 result
                    post_result = _wo.result
                    
                    # 报修前结果优先使用 initial_result_id
                    if hasattr(_wo, 'initial_result_id') and _wo.initial_result_id:
                        pre_result = _session.query(_DR).filter(_DR.id == _wo.initial_result_id).first()
                    
                    # 如果没有记录初始 ID，或者初始 ID 找不到，尝试通过逻辑找
                    if not pre_result:
                        # 如果任务还没到验收阶段，当前关联的其实就是初始结果
                        if not ("待审核" in cur_s or "已完成" in cur_s):
                            pre_result = post_result
                        else:
                            # 如果已验收，尝试找该管段在任务创建时的那个结果
                            if _wo.segment_id:
                                pre_result = _session.query(_DR).join(_DT).filter(
                                    _DT.segment_id == _wo.segment_id,
                                    _DR.created_at <= _wo.created_at
                                ).order_by(_DR.created_at.desc()).first()
                            
                            # 最后的兜底：如果还是没找到，找该任务创建前该管段的最早记录
                            if not pre_result and _wo.segment_id:
                                pre_result = _session.query(_DR).join(_DT).filter(
                                    _DT.segment_id == _wo.segment_id
                                ).order_by(_DR.created_at.asc()).first()

                            # 如果还是没有，尝试通过任务名称查找 (巡检发现问题: XXX)
                            if not pre_result:
                                task_name_prefix = f"巡检发现问题: {_wo.title.replace('巡检发现问题: ', '')}"
                                pre_result = _session.query(_DR).join(_DT).filter(
                                    _DT.task_name.like(f"%{task_name_prefix}%")
                                ).order_by(_DR.created_at.asc()).first()
                    
                    # 确定图片
                    pre_image = None
                    if pre_result:
                        # 尝试多种图片类型
                        for img_type in ['segmented', 'original', 'mask']:
                            pre_image = _session.query(_RI).filter(_RI.result_id == pre_result.id, _RI.image_type == img_type).first()
                            if pre_image: break
                        if not pre_image: pre_image = _session.query(_RI).filter(_RI.result_id == pre_result.id).first()
                        
                    post_image = None
                    # 只有当维修后结果存在，且不同于报修前结果时，才认为有“对比”
                    if post_result and (not pre_result or post_result.id != pre_result.id):
                        for img_type in ['segmented', 'original', 'mask']:
                            post_image = _session.query(_RI).filter(_RI.result_id == post_result.id, _RI.image_type == img_type).first()
                            if post_image: break
                        if not post_image: post_image = _session.query(_RI).filter(_RI.result_id == post_result.id).first()

                    # UI 展示
                    if post_image and pre_image:
                        # 双栏对比展示
                        comp_col1, comp_col2 = App.columns(2)
                        with comp_col1:
                            App.markdown("##### ⬅️ 报修前状态")
                            App.image(pre_image.image_data, use_container_width=True, caption=f"初始检出时间: {pre_result.created_at.strftime('%Y-%m-%d %H:%M')}")
                            if pre_result.deformation is not None:
                                App.caption(f"初始变形率: {pre_result.deformation:.2%}")
                        
                        with comp_col2:
                            App.markdown("##### ➡️ 维修后状态")
                            App.image(post_image.image_data, use_container_width=True, caption=f"验收提交时间: {post_result.created_at.strftime('%Y-%m-%d %H:%M')}")
                            
                            # 优先从备注中提取历史得分，保持一致性
                            import re
                            saved_score_match = re.search(r'AI\s*得分[:：]\s*(\d+)', _wo.processing_notes or "")
                            
                            # 使用统一的 HealthScore 计算评分
                            from HealthScore import calc_health_score
                            try:
                                res_json = json.loads(post_result.result_data)
                                # 优先从 JSON 中提取 RiskLevel，保持与 worker 端一致
                                risk_level = res_json.get("RiskLevel", ["中"])[0] if isinstance(res_json.get("RiskLevel"), list) else res_json.get("RiskLevel", "中")
                            except:
                                risk_level = "中"
                                if post_result.deformation is not None:
                                    th_high = float(db_ops.get_config("risk_threshold_high"))
                                    th_mid  = float(db_ops.get_config("risk_threshold_mid"))
                                    if post_result.deformation >= th_high: risk_level = "高"
                                    elif post_result.deformation < th_mid: risk_level = "低"
                            
                            mock_res = [{"Deformation": post_result.deformation or 0, "RiskLevel": risk_level}]
                            hs = calc_health_score(mock_res) # 不再传入硬编码参数，由内部获取
                            
                            # 如果有历史保存的分数，且计算出来的分数不一致，以历史分数为准（或者同时显示）
                            final_score = hs['score']
                            if saved_score_match:
                                saved_score = int(saved_score_match.group(1))
                                if saved_score != hs['score']:
                                    final_score = saved_score # 以工人提交时的分数为准
                            
                            App.metric("AI 验收评分", f"{final_score}/100", delta=f"{hs['level']}")
                            App.progress(final_score / 100)
                    elif pre_image:
                        # 单栏展示初始状态
                        App.markdown("##### 🔍 初始报修状态分析")
                        c1, c2 = App.columns([2, 1])
                        with c1:
                            App.image(pre_image.image_data, use_container_width=True, caption=f"检出时间: {pre_result.created_at.strftime('%Y-%m-%d %H:%M')}")
                        with c2:
                            if pre_result.deformation is not None:
                                App.metric("初始变形率", f"{pre_result.deformation:.2%}")
                            App.info("等待维修完成后进行 AI 验收对比")
                    elif post_image:
                        # 只有维修后状态（可能初始数据丢失）
                        App.markdown("##### 🔍 维修后验收分析")
                        c1, c2 = App.columns([2, 1])
                        with c1:
                            App.image(post_image.image_data, use_container_width=True, caption=f"验收时间: {post_result.created_at.strftime('%Y-%m-%d %H:%M')}")
                        with c2:
                            from HealthScore import calc_health_score
                            mock_res = [{"Deformation": post_result.deformation or 0, "RiskLevel": "中"}]
                            hs = calc_health_score(mock_res)
                            App.metric("AI 验收评分", f"{hs['score']}/100")
                            App.info("未找到报修前原始图像")
                    else:
                        App.warning("⚠️ 暂未找到该任务的关联图像数据")

                    # 强制通过说明补充
                    if hasattr(_wo, 'is_forced') and _wo.is_forced:
                        with App.container(border=True):
                            App.warning(f"⚠️ **强制核准说明**:\n{_wo.processing_notes.split('【强制核准】原因：')[-1] if '【强制核准】原因：' in _wo.processing_notes else '未填写'}")
                elif _wo:
                    App.info("该任务暂无关联的 AI 分析图像")
            finally:
                workorder_service.db_manager.close_session(_session)
        except Exception as e:
            App.error(f"加载任务详情失败: {str(e)}")


        update_col1, update_col2 = App.columns(2)
        with update_col1:
            status_options = ["待指派", "进行中", "待审核", "已完成"]
            # 提取原始状态：去掉表情符号、中括号和额外说明
            import re
            raw_cur_status = re.sub(r'[🚨⚠️\[\]\s(评分预警)(特批)]', '', cur_s) if cur_s else ""
            
            # 如果提取后不在选项中，尝试保守提取
            if raw_cur_status not in status_options:
                for opt in status_options:
                    if opt in cur_s:
                        raw_cur_status = opt
                        break
            
            # 定义逻辑下一步映射
            next_step_map = {
                "待指派": "进行中",
                "进行中": "待审核",
                "待审核": "已完成",
                "已完成": "已完成"
            }
            recommended_next = next_step_map.get(raw_cur_status, raw_cur_status)
            
            # 计算默认选中的索引（优先呈现下一步）
            try:
                default_idx = status_options.index(recommended_next)
            except ValueError:
                default_idx = status_options.index(raw_cur_status) if raw_cur_status in status_options else 0

            App.markdown(f"📍 **当前状态**: `{raw_cur_status}` → 💡 **建议下一步**: `{recommended_next}`")
            next_status = App.selectbox(
                "变更状态 (默认已选中推荐步骤)",
                status_options,
                index=default_idx,
                key="workorder_next_status",
            )
        with update_col2:
            if user_info.get("role") == "admin":
                # 管理员可以从下拉框选择工人
                all_workers = user_service.list_users(role="worker", status="active")
                worker_options = [w.username for w in all_workers]
                worker_display = {w.username: (f"{w.full_name} ({w.username})" if w.full_name else w.username) for w in all_workers}
                
                if not worker_options:
                    App.warning("系统中暂无可用工人账号")
                    assignee = App.text_input("负责人 (手动输入)", key="workorder_assignee")
                else:
                    # 默认选中当前负责人，如果不在列表中则置空
                    try:
                        cur_idx = worker_options.index(current_order.get("负责人"))
                    except ValueError:
                        cur_idx = 0
                    assignee = App.selectbox(
                        "指派负责人", 
                        worker_options, 
                        index=cur_idx, 
                        format_func=lambda x: worker_display.get(x, x),
                        key="workorder_assignee_sel"
                    )
            else:
                # 工人端固定为自己
                assignee = App.text_input("负责人", key="workorder_assignee", disabled=True)

        notes = App.text_area("处理记录", key="workorder_notes", placeholder="填写派单、维修、复检等过程说明")
        
        btn_col1, btn_col2 = App.columns([3, 1])
        with btn_col1:
            if App.button("💾 保存任务更新", key="workorder_update_btn", type="primary", use_container_width=True):
                # 如果是管理员使用了下拉框，则取下拉框的值
                final_assignee = assignee
                if user_info.get("role") == "admin" and 'workorder_assignee_sel' in App.session_state:
                    final_assignee = App.session_state['workorder_assignee_sel']
                
                updated = workorder_service.update_work_order(
                    selected_order_id,
                    status=next_status,
                    assignee=final_assignee,
                    processing_notes=notes,
                )
                if updated:
                    audit_service.log_action(user_info["id"], user_info["username"], "更新任务", "任务管理", 
                                           f"更新了任务 {current_order['任务编号']}：状态->{next_status}，负责人->{final_assignee}")
                    App.toast("✅ 任务已更新")
                    
                    # 清理输入状态，以便在 rerun 后从数据库重新加载最新数据
                    for k in ["workorder_notes", "workorder_next_status", "workorder_assignee_sel"]:
                        if k in App.session_state:
                            del App.session_state[k]
                    
                    App.rerun()
                else:
                    App.warning("任务不存在或更新失败")
        
        with btn_col2:
            if user_info.get("role") == "admin":
                if App.button("🗑️ 删除", key="workorder_single_del_btn", type="secondary", use_container_width=True):
                    if workorder_service.delete_work_orders([selected_order_id]):
                        audit_service.log_action(user_info["id"], user_info["username"], "删除任务", "任务管理", 
                                               f"删除了任务 {current_order['任务编号']}")
                        App.toast("🗑️ 任务已删除")
                        App.rerun()

        App.markdown("#### 🧾 流转日志")
        # 默认只显示最近 10 条，并提供按钮加载更多
        log_limit = App.session_state.get("workorder_log_limit", 10)
        logs = workorder_service.list_work_order_logs(selected_order_id, limit=log_limit)
        
        if logs:
            for log in logs:
                with App.container(border=True):
                    l_c1, l_c2 = App.columns([1, 4])
                    with l_c1:
                        App.caption(log.created_at.strftime('%m-%d %H:%M') if log.created_at else "-")
                        App.info(log.action_type)
                    with l_c2:
                        App.markdown(f"**任务编号:** `{log.order_code or '-'}` | **状态变动:** `{log.old_status or '-'}` ➔ `{log.new_status or '-'}`")
                        if log.assignee:
                            App.markdown(f"**负责人:** {log.assignee}")
                        if log.notes:
                            App.markdown(f"**操作说明:** {log.notes}")
            
            # 如果记录较多，显示加载更多按钮
            if len(logs) >= log_limit:
                if App.button("📂 查看更多历史记录", key="wo_load_more_logs"):
                    App.session_state["workorder_log_limit"] = log_limit + 20
                    App.rerun()
        else:
            App.info("暂无流转记录")


@App.fragment
def show_worker_task_center(worker_id):
    """工人任务中心：今日任务清单"""
    user_info = App.session_state.get("user_info", {})
    App.markdown(f"### 📋 您好，{user_info.get('full_name') or user_info.get('username', '用户')}")
    App.markdown("这是您当前的待办任务清单。")

    # 获取该工人的任务
    my_tasks = workorder_service.list_work_orders(status="全部")
    username = user_info.get("username")
    if username:
        # 只看待安排、正在修、等验收的任务
        my_tasks = [t for t in my_tasks if t.assignee == username and t.status in ["待指派", "进行中", "待审核"]]
    else:
        my_tasks = []
    
    # 统计信息
    pending_count = len([t for t in my_tasks if t.status == "待指派"])
    processing_count = len([t for t in my_tasks if t.status == "进行中"])
    audit_count = len([t for t in my_tasks if t.status == "待审核"])

    st_col1, st_col2, st_col3, st_col4 = App.columns(4)
    st_col1.metric("待处理", pending_count)
    st_col2.metric("处理中", processing_count)
    st_col3.metric("待验收", audit_count)
    st_col4.metric("紧急任务", len([t for t in my_tasks if t.risk_level == "高"]), delta_color="inverse")
    
    if not my_tasks:
        App.success("✨ 当前所有待办任务已处理完毕。")
        return

    _session = workorder_service.db_manager.get_session()
    try:
        from database import ResultImage as _RI
        for t in my_tasks:
            with App.container(border=True):
                c_info, c_action = App.columns([4, 1.5])
                with c_info:
                    risk_emoji = "🔴" if t.risk_level == "高" else "🟡" if t.risk_level == "中" else "🟢"
                    
                    status_display = "⚪ 待安排"
                    if t.status == "进行中":
                        status_display = "🟡 处理中"
                    elif t.status == "待审核":
                        status_display = "🔵 待验收"
                    
                    App.markdown(f"**{risk_emoji} {t.order_code}** | {status_display}")
                    App.markdown(f"#### {t.title}")
                    App.caption(f"📍 位置: {t.segment.location if t.segment else '未知'} | 🛠️ 管段: {t.segment.segment_code if t.segment else '-'}")
                    
                    if t.status == "处理中" and "【管理员驳回】" in (t.processing_notes or ""):
                        import re
                        reject_notes = re.findall(r'建议：(.*?)(?:\n|$)', t.processing_notes)
                        if reject_notes:
                            App.error(f"❌ **审核未通过，处理意见**: {reject_notes[-1]}")
                    
                    with App.expander("📝 任务详细说明"):
                        App.write(t.issue_description)
                        # 显示报修前的图片供工人参考
                        target_res_id = t.initial_result_id or t.result_id
                        if target_res_id:
                            img_obj = _session.query(_RI).filter(_RI.result_id == target_res_id).first()
                            if img_obj:
                                App.image(img_obj.image_data, caption="📸 报修前现场图片 (参考)", use_container_width=True)
                            else:
                                App.caption("💡 暂无原始现场图片")
                        else:
                            App.caption("💡 暂无原始现场图片")
                    
                    if t.processing_notes:
                        with App.expander("💬 历史记录"):
                            App.caption(t.processing_notes)
                    
                with c_action:
                    App.write("") # 占位
                    if t.status in ["待指派", "进行中"]:
                        if App.button("✅ 我修好了，拍照验收", key=f"verify_task_{t.id}", type="primary", use_container_width=True):
                            if t.status == "待指派":
                                workorder_service.update_work_order(t.id, status="进行中", processing_notes="已开始执行维护任务。")
                                audit_service.log_action(user_info["id"], user_info["username"], "自动开工", "任务中心", f"任务 {t.order_code} 自动开始并进入验收")
                            
                            App.session_state["active_wo_for_detect"] = {
                                "id": t.id,
                                "code": t.order_code,
                                "segment": t.segment.segment_code if t.segment else "未知",
                                "task_id": t.task_id
                            }
                            App.session_state["WORKER_SCAN_RES"] = None
                            App.session_state["active_tab"] = "📸 拍照验收"
                            App.rerun()
                        
                        if t.status == "待指派":
                            if App.button("▶️ 标记开工", key=f"start_task_{t.id}", type="secondary", use_container_width=True):
                                workorder_service.update_work_order(t.id, status="进行中", processing_notes="作业人员已到达现场并开始维护作业。")
                                audit_service.log_action(user_info["id"], user_info["username"], "标记开工", "任务中心", f"开始执行任务 {t.order_code}")
                                App.rerun()
    finally:
        workorder_service.db_manager.close_session(_session)

def show_report_center():
    """统一报表中心"""
    try:
        App.markdown("### 📊 报表中心")
        App.caption("支持批量导出巡检分析报告与任务汇总表")

        with App.container(border=True):
            col1, col2, col3 = App.columns(3)
            with col1:
                date_range = App.date_input("选择时间范围", [datetime.now().replace(day=1), datetime.now()])
            with col2:
                report_type = App.selectbox("报表类型", ["巡检月度分析报告", "任务验收汇总表", "风险隐患清单"])
            with col3:
                export_format = App.selectbox("导出格式", ["Excel", "PDF"], index=0)

        filter_col1, filter_col2 = App.columns(2)
        with filter_col1:
            risk_filter = App.multiselect("风险等级过滤", ["高", "中", "低"], default=["高", "中"])
        with filter_col2:
            segments = segment_service.list_segments()
            seg_options = {0: "全部区域"}
            seg_options.update({s.id: f"{s.segment_code} ({s.location or '未知'})" for s in segments})
            area_filter = App.selectbox("所属区域/管段", list(seg_options.keys()), format_func=lambda x: seg_options[x])

        App.markdown("---")
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            # 实际数据检索
            from db_service import detection_service, workorder_service
            
            # 获取巡检记录
            all_results = detection_service.list_results(limit=1000)
            filtered_results = [
                r for r in all_results 
                if r.created_at and start_date <= r.created_at.date() <= end_date
                and (not risk_filter or r.risk_level in risk_filter)
                and (area_filter == 0 or (getattr(r, 'task', None) and getattr(r.task, 'segment_id', None) == area_filter))
            ]
            
            # 获取任务记录
            all_orders = workorder_service.list_work_orders(status="全部")
            filtered_orders = [
                o for o in all_orders
                if o.created_at and start_date <= o.created_at.date() <= end_date
                and (not risk_filter or o.risk_level in risk_filter)
                and (area_filter == 0 or o.segment_id == area_filter)
            ]
            
            App.info(f"🔍 已检索到 {start_date} 至 {end_date} 期间符合条件的数据：{len(filtered_results)} 条巡检记录，{len(filtered_orders)} 条任务记录。")
            
            if not filtered_results and not filtered_orders:
                App.warning("该时间段或过滤条件下暂无数据。")
                return

            # 数据预览
            with App.expander("👁️ 待导出数据预览", expanded=False):
                if report_type == "任务验收汇总表":
                    preview_data = []
                    for o in filtered_orders:
                        preview_data.append({
                            "任务编号": getattr(o, 'order_code', '未知'), 
                            "标题": getattr(o, 'title', '无标题'), 
                            "风险等级": getattr(o, 'risk_level', '未知'), 
                            "负责人": getattr(o, 'assignee', '未指派'), 
                            "状态": getattr(o, 'status', '未知'), 
                            "创建日期": o.created_at.strftime("%Y-%m-%d") if o.created_at else "-"
                        })
                else:
                    preview_data = []
                    for r in filtered_results:
                        seg_code = "未知"
                        if r.task and r.task.segment:
                            seg_code = r.task.segment.segment_code
                        
                        preview_data.append({
                            "记录ID": r.id, 
                            "变形率": f"{r.deformation:.2%}" if r.deformation else "-", 
                            "风险": getattr(r, 'risk_level', '未知'), 
                            "所属管段": seg_code,
                            "分析日期": r.created_at.strftime("%Y-%m-%d") if r.created_at else "-"
                        })
                if preview_data:
                    App.dataframe(preview_data, use_container_width=True, hide_index=True)
                else:
                    App.info("预览数据为空")

            c1, c2, c3 = App.columns([2, 2, 1])
            with c2:
                if App.button(f"🚀 批量生成并下载 {report_type}", use_container_width=True, type="primary"):
                    with App.spinner("正在汇总数据并生成报表..."):
                        from Export import GetImageExcel, GetWorkOrderExcel
                        
                        user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
                        audit_service.log_action(user_info["id"], user_info["username"], "生成报表", "报表中心", f"生成了 {report_type} ({export_format})")
                        
                        if report_type == "任务验收汇总表":
                            export_data = []
                            for o in filtered_orders:
                                export_data.append({
                                    "任务编号": getattr(o, 'order_code', '未知'),
                                    "标题": getattr(o, 'title', '无标题'),
                                    "风险等级": getattr(o, 'risk_level', '未知'),
                                    "优先级": getattr(o, 'priority', '中'),
                                    "状态": getattr(o, 'status', '未知'),
                                    "负责人": getattr(o, 'assignee', '未指派'),
                                    "处理记录": getattr(o, 'processing_notes', ''),
                                    "创建时间": o.created_at.strftime("%Y-%m-%d %H:%M:%S") if o.created_at else "-"
                                })
                            
                            if export_format == "Excel":
                                from Export import GetTableExcel
                                buf = GetTableExcel(export_data, Title="任务验收汇总")
                            else:
                                from Export import GetWorkOrderPDF
                                buf = GetWorkOrderPDF(export_data)
                        else:
                            # 转换巡检记录数据 (用于 巡检月度分析报告 或 风险隐患清单)
                            export_data = []
                            for r in filtered_results:
                                try:
                                    res_json = json.loads(r.result_data) if r.result_data else {}
                                except: res_json = {}
                                
                                # 确保获取标量值，防止 AI 推理结果以列表形式存储导致 Excel 导出失败
                                def get_scalar(d, key, default_val):
                                    val = d.get(key, default_val)
                                    if isinstance(val, list):
                                        return val[0] if val else default_val
                                    return val

                                seg_code = "未知"
                                if r.task and r.task.segment:
                                    seg_code = r.task.segment.segment_code

                                export_data.append({
                                    "记录ID": r.id,
                                    "变形率": f"{r.deformation:.2%}" if r.deformation else "0.00%",
                                    "风险等级": getattr(r, 'risk_level', '未知'),
                                    "所属管段": seg_code,
                                    "分析日期": r.created_at.strftime("%Y-%m-%d") if r.created_at else "-",
                                    "形状特征": get_scalar(res_json, "Shape", "-"),
                                    "处理建议": get_scalar(res_json, "ActionSuggestion", "-")
                                })
                            
                            if export_format == "Excel":
                                from Export import GetTableExcel
                                buf = GetTableExcel(export_data, Title=report_type)
                            else:
                                from Export import GetImagePDF
                                buf = GetImagePDF(export_data, meta={
                                    "project": "下水道管网巡检月度分析报告" if report_type == "巡检月度分析报告" else "风险隐患清单",
                                    "report_type": report_type,
                                    "date": f"{start_date} ~ {end_date}"
                                })
                        
                        App.success("✅ 报表生成成功！")
                        file_ext = "xlsx" if export_format == "Excel" else "pdf"
                        App.download_button(
                            label="📥 点击下载报表",
                            data=buf.getvalue(),
                            file_name=f"{report_type}_{datetime.now().strftime('%Y%m%d')}.{file_ext}",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_format == "Excel" else "application/pdf"
                        )
    except Exception as e:
        import traceback
        App.error(f"❌ 报表中心内部错误: {str(e)}")
        App.code(traceback.format_exc())

def show_settings_dashboard():
    """系统设置页"""
    App.markdown("### ⚙️ 系统设置")

    # ── 用户与审计管理 (仅管理员可见) ───────────────────────
    user_info = App.session_state.get("user_info", {})
    if user_info.get("role") == "admin":
        with App.expander("👥 账号管理", expanded=True):
            tab1, tab2, tab3 = App.columns(3)
            with tab1:
                view_mode = "查看账号"
            with tab2:
                add_mode = "新增账号"
            with tab3:
                edit_mode = "编辑账号"
            
            active_tab = App.radio("操作类型", ["查看账号", "新增账号", "编辑账号"], horizontal=True, key="user_mgmt_tab")
            
            users = user_service.list_users()
            
            if active_tab == "查看账号":
                App.markdown("##### 📋 账号列表")
                if users:
                    user_rows = []
                    for u in users:
                        status_text = "正常" if u.status == "active" else "待审批" if u.status == "pending" else "禁用"
                        role_text = "管理员" if u.role == "admin" else "作业人员"
                        user_rows.append({
                            "用户名": u.username,
                            "角色": role_text,
                            "状态": status_text,
                            "最后登录": u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "从未登录"
                        })
                    App.dataframe(user_rows, hide_index=True, use_container_width=True)
                    
                    pending_users = [u for u in users if u.status == "pending"]
                    if pending_users:
                        App.markdown("##### ⏳ 待审批申请")
                        for pu in pending_users:
                            pc1, pc2 = App.columns([3, 1])
                            with pc1:
                                App.write(f"**{pu.username}** ({pu.email or '无邮箱'})")
                            with pc2:
                                pcol1, pcol2 = App.columns(2)
                                if pcol1.button("通过", key=f"apr_{pu.id}"):
                                    user_service.update_user_status(pu.id, "active")
                                    audit_service.log_action(user_info["id"], user_info["username"], "审批", "账号管理", f"批准账号 {pu.username}")
                                    App.success(f"已批准 {pu.username}")
                                    App.rerun()
                                if pcol2.button("拒绝", key=f"rej_{pu.id}"):
                                    user_service.delete_user(pu.id)
                                    audit_service.log_action(user_info["id"], user_info["username"], "审批", "账号管理", f"拒绝账号 {pu.username}")
                                    App.rerun()
                else:
                    App.info("暂无账号记录")
            
            elif active_tab == "新增账号":
                App.markdown("##### ➕ 新增账号")
                new_username = App.text_input("用户名（唯一）", placeholder="请输入用户名", key="new_user_name")
                new_email = App.text_input("电子邮箱", placeholder="可选", key="new_user_email")
                new_role = App.selectbox("账号角色", ["user", "admin"], format_func=lambda x: "管理员" if x == "admin" else "作业人员", key="new_user_role")
                new_password = App.text_input("初始密码", type="password", placeholder="请输入密码", key="new_user_pass")
                new_password2 = App.text_input("确认密码", type="password", placeholder="请再次输入密码", key="new_user_pass2")
                
                if App.button("创建账号", key="create_user_btn", type="primary", use_container_width=True):
                    if not new_username.strip():
                        App.error("用户名不能为空")
                    elif not new_password:
                        App.error("密码不能为空")
                    elif new_password != new_password2:
                        App.error("两次密码输入不一致")
                    else:
                        try:
                            user_service.create_user(
                                username=new_username.strip(),
                                email=new_email.strip() or None,
                                password_hash=new_password,
                                role=new_role
                            )
                            audit_service.log_action(user_info["id"], user_info["username"], "新增", "账号管理", f"新增账号 {new_username}（{new_role}）")
                            App.success(f"账号 {new_username} 创建成功")
                            App.rerun()
                        except Exception as e:
                            App.error(f"创建失败: {e}")
            
            else:
                App.markdown("##### ✏️ 编辑账号")
                if not users:
                    App.info("暂无账号可编辑")
                else:
                    edit_user_id = App.selectbox("选择要编辑的账号", [u.id for u in users],
                                                  format_func=lambda uid: next((u.username for u in users if u.id == uid), ""),
                                                  key="edit_user_select")
                    cur_user = next((u for u in users if u.id == edit_user_id), None)
                    if cur_user:
                        App.markdown(f"当前编辑: **{cur_user.username}**")
                        e_email = App.text_input("电子邮箱", value=cur_user.email or "", placeholder="可选", key=f"edit_email_{edit_user_id}")
                        e_role = App.selectbox("账号角色", ["user", "admin"],
                                               index=0 if cur_user.role == "user" else 1,
                                               format_func=lambda x: "管理员" if x == "admin" else "作业人员",
                                               key=f"edit_role_{edit_user_id}")
                        e_status = App.selectbox("账号状态", ["active", "pending", "disabled"],
                                                  index=0 if cur_user.status == "active" else 1 if cur_user.status == "pending" else 2,
                                                  format_func=lambda x: "正常" if x == "active" else "待审批" if x == "pending" else "禁用",
                                                  key=f"edit_status_{edit_user_id}")
                        
                        reset_pw = App.checkbox("重置密码", key=f"reset_pw_{edit_user_id}")
                        if reset_pw:
                            new_pw = App.text_input("新密码", type="password", placeholder="请输入新密码", key=f"new_pw_{edit_user_id}")
                        
                        del_user = App.checkbox(f"删除账号 {cur_user.username}（不可恢复）", key=f"del_user_{edit_user_id}")
                        
                        c1, c2 = App.columns(2)
                        with c1:
                            if App.button("保存修改", key=f"save_edit_{edit_user_id}", type="primary", use_container_width=True):
                                try:
                                    update_data = {"email": e_email.strip() or None, "role": e_role, "status": e_status}
                                    if reset_pw and new_pw:
                                        update_data["password_hash"] = new_pw
                                    user_service.update_user(edit_user_id, **update_data)
                                    audit_service.log_action(user_info["id"], user_info["username"], "修改", "账号管理", f"修改账号 {cur_user.username}")
                                    App.success(f"账号 {cur_user.username} 已更新")
                                    App.rerun()
                                except Exception as e:
                                    App.error(f"保存失败: {e}")
                        with c2:
                            if del_user:
                                if App.button("确认删除", key=f"confirm_del_{edit_user_id}", type="primary", use_container_width=True):
                                    user_service.delete_user(edit_user_id)
                                    audit_service.log_action(user_info["id"], user_info["username"], "删除", "账号管理", f"删除账号 {cur_user.username}")
                                    App.success(f"账号 {cur_user.username} 已删除")
                                    App.rerun()

        with App.expander("📜 系统审计日志", expanded=False):
            logs = audit_service.list_logs(limit=200)
            if logs:
                log_rows = []
                for l in logs:
                    log_rows.append({
                        "时间": l.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "操作人": l.username,
                        "动作": l.action,
                        "模块": l.module,
                        "内容": l.content
                    })
                App.dataframe(log_rows, hide_index=True, use_container_width=True)
            else:
                App.info("暂无审计日志")

        with App.expander("🚀 AI 推理加速与模型管理", expanded=False):
            App.markdown("#### 🛠️ 模型加速管理")
            App.caption("将原始 PyTorch (.pt) 模型导出为 ONNX 或 TensorRT 格式，可大幅提升推理速度。")
            
            # 检查环境
            env_col1, env_col2 = App.columns(2)
            with env_col1:
                try:
                    import onnxruntime
                    providers = onnxruntime.get_available_providers()
                    if 'CUDAExecutionProvider' in providers:
                        App.success("✅ ONNX Runtime GPU (CUDA) 已就绪")
                    else:
                        App.warning("⚠️ ONNX Runtime 运行在 CPU 模式")
                        App.info("建议安装: `pip install onnxruntime-gpu` 以启用 GPU 加速")
                except ImportError:
                    App.error("❌ 未安装 onnxruntime")
                    App.info("建议安装: `pip install onnxruntime-gpu` 或 `onnxruntime`")
            
            with env_col2:
                try:
                    import torch as Torch
                    if Torch.cuda.is_available():
                        App.success(f"✅ CUDA GPU 可用: {Torch.cuda.get_device_name(0)}")
                    else:
                        App.info("ℹ️ 当前环境仅支持 CPU 推理")
                except ImportError:
                    App.error("❌ 未安装 torch")

            # 模型列表与导出
            model_dir = OS.path.join(OS.path.dirname(OS.path.dirname(__file__)), "Models")
            if OS.path.exists(model_dir):
                models = [f for f in OS.listdir(model_dir) if f.endswith(".pt")]
                if models:
                    selected_m = App.selectbox("选择要加速的模型", models, key="acc_model_select")
                    target_fmt = App.radio("目标加速格式", ["ONNX (通用加速)", "Engine (TensorRT - 仅限 NVIDIA GPU)"], horizontal=True)
                    fmt_ext = "onnx" if "ONNX" in target_fmt else "engine"
                    
                    if App.button(f"🚀 开始导出并优化 {fmt_ext.upper()}", use_container_width=True):
                        with App.spinner(f"正在转换模型 {selected_m} 至 {fmt_ext.upper()}... 这可能需要几分钟时间"):
                            from Caching import ExportModel
                            full_path = OS.path.join(model_dir, selected_m)
                            exported = ExportModel(full_path, fmt_ext)
                            if exported:
                                audit_service.log_action(user_info["id"], user_info["username"], "模型导出", "模型管理", f"导出模型 {selected_m} 为 {fmt_ext}")
                                App.success(f"✅ 导出成功！文件位置: {exported}")
                                App.info("提示：系统在下次加载模型时会自动优先选择加速后的版本。")
                else:
                    App.info("Models 目录下暂无 .pt 模型文件")
            else:
                App.error(f"找不到模型目录: {model_dir}")

        App.markdown("---")

    # ── 风险阈值配置 ──────────────────────────────────────────
    App.markdown("#### 🎚️ 风险阈值配置")
    App.caption("变形率（Deformation）范围 0~1，值越大表示变形越严重")

    cur_high = float(db_ops.get_config("risk_threshold_high", 0.3))
    cur_mid  = float(db_ops.get_config("risk_threshold_mid",  0.15))

    th_col1, th_col2 = App.columns(2)
    with th_col1:
        new_high = App.slider(
            "高风险阈值（变形率 ≥ 此值 → 高风险）",
            min_value=0.05, max_value=1.0, value=cur_high, step=0.01,
            key="setting_risk_high",
        )
    with th_col2:
        new_mid = App.slider(
            "中风险阈值（变形率 ≥ 此值 → 中风险，低于此值 → 低风险）",
            min_value=0.01, max_value=1.0, value=cur_mid, step=0.01,
            key="setting_risk_mid",
        )

    if new_high <= new_mid:
        App.warning("高风险阈值必须大于中风险阈值")
    else:
        App.info(f"当前规则：变形率 ≥ {new_high:.2f} → 高风险 ｜ {new_mid:.2f} ~ {new_high:.2f} → 中风险 ｜ < {new_mid:.2f} → 低风险")
        if App.button("💾 保存风险阈值", key="setting_save_threshold"):
            db_ops.set_config("risk_threshold_high", new_high, "number", "高风险变形率阈值")
            db_ops.set_config("risk_threshold_mid",  new_mid,  "number", "中风险变形率阈值")
            # 添加审计日志
            user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
            audit_service.log_action(user_info["id"], user_info["username"], "修改配置", "风险设置", f"更新风险阈值: 高={new_high}, 中={new_mid}")
            App.success("风险阈值已保存")

    App.markdown("---")

    # ── 健康评分权重配置 ──────────────────────────────────────
    App.markdown("#### ⚖️ 健康评分权重配置")
    App.caption("调整不同风险等级对最终健康得分的影响权重")

    cur_w_high = float(db_ops.get_config("health_penalty_high", 1.5))
    cur_w_mid  = float(db_ops.get_config("health_penalty_mid",  0.5))
    cur_l_high = float(db_ops.get_config("health_limit_high",   60.0))
    cur_l_mid  = float(db_ops.get_config("health_limit_mid",    25.0))

    w_col1, w_col2 = App.columns(2)
    with w_col1:
        new_w_high = App.number_input("高风险帧扣分系数 (每1%)", value=cur_w_high, step=0.1, key="setting_w_high")
        new_l_high = App.number_input("高风险扣分上限", value=cur_l_high, step=1.0, key="setting_l_high")
    with w_col2:
        new_w_mid = App.number_input("中风险帧扣分系数 (每1%)", value=cur_w_mid, step=0.1, key="setting_w_mid")
        new_l_mid = App.number_input("中风险扣分上限", value=cur_l_mid, step=1.0, key="setting_l_mid")

    if App.button("💾 保存权重配置", key="setting_save_weights"):
        db_ops.set_config("health_penalty_high", new_w_high, "number", "高风险帧扣分系数")
        db_ops.set_config("health_penalty_mid",  new_w_mid,  "number", "中风险帧扣分系数")
        db_ops.set_config("health_limit_high",   new_l_high, "number", "高风险扣分上限")
        db_ops.set_config("health_limit_mid",    new_l_mid,  "number", "中风险扣分上限")
        # 添加审计日志
        user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
        audit_service.log_action(user_info["id"], user_info["username"], "修改配置", "评分设置", f"更新评分权重: 高权重={new_w_high}, 中权重={new_w_mid}")
        App.success("评分权重配置已保存")

    App.markdown("---")

    # ── 其他检测设置 ──────────────────────────────────────────
    App.markdown("#### 🔧 检测参数配置")

    cur_batch    = int(db_ops.get_config("batch_size_default", 5))
    cur_spacing  = int(db_ops.get_config("frame_spacing_default", 30))
    cur_mode     = str(db_ops.get_config("processing_mode_default", "逐帧"))

    p_col1, p_col2, p_col3 = App.columns(3)
    with p_col1:
        new_batch = App.number_input("默认每批处理条数", min_value=1, max_value=50, value=cur_batch, step=1, key="setting_batch")
    with p_col2:
        new_spacing = App.number_input("默认采样间隔（帧）", min_value=1, max_value=300, value=cur_spacing, step=1, key="setting_spacing")
    with p_col3:
        mode_options = ["逐帧", "间隔采样"]
        new_mode = App.selectbox("默认分析模式", mode_options,
            index=mode_options.index(cur_mode) if cur_mode in mode_options else 0,
            key="setting_mode")

    if App.button("💾 保存检测参数", key="setting_save_params"):
        db_ops.set_config("batch_size_default",      new_batch,   "number", "默认处理数量")
        db_ops.set_config("frame_spacing_default",   new_spacing, "number", "默认采样间隔")
        db_ops.set_config("processing_mode_default", new_mode,    "string", "默认处理模式")
        user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
        audit_service.log_action(user_info["id"], user_info["username"], "修改配置", "检测参数", f"处理数量={new_batch}, 采样间隔={new_spacing}, 模式={new_mode}")
        App.success("检测参数已保存")

    App.markdown("---")

    # ── 数据清理 ──────────────────────────────────────────────
    App.markdown("#### 🗑️ 历史数据清理")
    App.caption("清除指定天数之前的检测记录、分析结果及图像数据（维修任务数据不受影响）")

    clean_col1, clean_col2 = App.columns([2, 1])
    with clean_col1:
        days = App.slider("清除多少天前的数据", min_value=1, max_value=365, value=30, step=1, key="setting_clean_days")
        App.caption(f"将删除 {days} 天前创建的所有检测任务及其关联结果和图像")
    with clean_col2:
        App.markdown("<br>", unsafe_allow_html=True)
        confirm_clean = App.checkbox("确认清理（不可恢复）", key="setting_clean_confirm")
        if confirm_clean:
            if App.button("🗑️ 执行清理", key="setting_clean_btn", type="primary", use_container_width=True):
                with App.spinner("正在清理..."):
                    result = db_integration.cleanup_old_data(days)
                if result:
                    App.success(
                        f"清理完成：删除任务 {result['deleted_tasks']} 条，"
                        f"结果 {result['deleted_results']} 条，"
                        f"图像 {result['deleted_images']} 张"
                    )
                    App.session_state.pop("setting_clean_confirm", None)
                else:
                    App.info("没有符合条件的历史数据")

    App.markdown("---")

    # ── 当前配置总览 ──────────────────────────────────────────
    with App.expander("📋 当前所有配置项", expanded=False):
        try:
            from database import SystemConfig
            session = db_ops.db_manager.get_session()
            try:
                configs = session.query(SystemConfig).filter(SystemConfig.is_active == True).all()
                rows = [{"配置键": c.config_key, "值": c.config_value, "类型": c.config_type, "说明": c.description or ""} for c in configs]
                App.dataframe(rows, hide_index=True, use_container_width=True)
            finally:
                db_ops.db_manager.close_session(session)
        except Exception as e:
            App.error(f"读取配置失败: {e}")


@App.fragment
def show_pipeline_ledger():
    """管网电子档案管理页"""
    App.markdown("### 🗂️ 管网电子档案")
    
    user_info = App.session_state.get("user_info", {})

    segments = segment_service.list_segments()

    # ── 辅助：将任务 ID 列表压缩为范围字符串，如 [1,2,3,5,7,8] → "1-3, 5, 7-8" ──
    def _ids_to_range_str(ids):
        if not ids:
            return "—"
        ids = sorted(set(ids))
        parts, start, end = [], ids[0], ids[0]
        for i in ids[1:]:
            if i == end + 1:
                end = i
            else:
                parts.append(str(start) if start == end else f"{start}-{end}")
                start = end = i
        parts.append(str(start) if start == end else f"{start}-{end}")
        return ", ".join(parts)

    # ── 台账列表 ──────────────────────────────────────────────
    if segments:
        rows = []
        for s in segments:
            rows.append({
                "选择": False,
                "id": s.id,
                "管段编号": s.segment_code,
                "位置": s.location or "",
                "管径(mm)": s.diameter,
                "材质": s.material or "",
                "长度(m)": s.length,
                "安装年份": s.install_year,
                "上次检测": s.last_inspected_at.strftime('%Y-%m-%d') if s.last_inspected_at else "未检测",
                "备注": s.notes or "",
            })

        edited = App.data_editor(
            rows,
            column_config={
                "选择": App.column_config.CheckboxColumn("选择", default=False),
                "id": App.column_config.NumberColumn("ID", disabled=True),
                "管段编号": App.column_config.TextColumn("管段编号", disabled=True),
                "位置": App.column_config.TextColumn("位置", disabled=True),
                "管径(mm)": App.column_config.NumberColumn("管径(mm)", disabled=True),
                "材质": App.column_config.TextColumn("材质", disabled=True),
                "长度(m)": App.column_config.NumberColumn("长度(m)", disabled=True),
                "安装年份": App.column_config.NumberColumn("安装年份", disabled=True),
                "上次检测": App.column_config.TextColumn("上次检测", disabled=True),
                "备注": App.column_config.TextColumn("备注", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="segment_table_editor",
        )

        selected = [r for r in edited if r.get("选择")]
        if selected:
            App.caption(f"已选 {len(selected)} 条管段")
            # 只有管理员可以删除
            if user_info.get("role") == "admin":
                del_confirm = App.checkbox("确认删除所选管段（关联任务的管段信息将被清除）", key="seg_del_confirm")
                if del_confirm and App.button("🗑️ 删除所选管段", key="seg_del_btn", type="primary"):
                    for r in selected:
                        segment_service.delete_segment(r["id"])
                        audit_service.log_action(user_info["id"], user_info["username"], "删除", "管段台账", f"删除了管段 ID:{r['id']}")
                    App.success(f"已删除 {len(selected)} 条管段")
                    App.session_state.pop("seg_del_confirm", None)
                    App.rerun()
            else:
                App.info("💡 只有管理员有权删除管段。")
    else:
        App.info("暂无管段记录，请在下方新增")

    App.markdown("---")

    # ── 编辑已有管段 (仅管理员) ──────────────────────────────────────────
    if segments and user_info.get("role") == "admin":
        App.markdown("#### ✏️ 编辑管段信息")
        seg_options = {s.id: f"{s.segment_code} — {s.location or '无位置'}" for s in segments}
        edit_id = App.selectbox("选择要编辑的管段", list(seg_options.keys()),
                                format_func=lambda x: seg_options[x], key="seg_edit_select")
        cur = next((s for s in segments if s.id == edit_id), None)
        if cur:
            ec1, ec2, ec3 = App.columns(3)
            with ec1:
                e_loc = App.text_input("位置", value=cur.location or "", key=f"seg_e_loc_{edit_id}")
                e_mat = App.text_input("材质", value=cur.material or "", key=f"seg_e_mat_{edit_id}")
            with ec2:
                e_dia = App.number_input("管径(mm)", value=float(cur.diameter or 0), min_value=0.0, step=1.0, key=f"seg_e_dia_{edit_id}")
                e_len = App.number_input("长度(m)", value=float(cur.length or 0), min_value=0.0, step=0.1, key=f"seg_e_len_{edit_id}")
            with ec3:
                e_year = App.number_input("安装年份", value=int(cur.install_year or 2000), min_value=1900, max_value=2100, step=1, key=f"seg_e_year_{edit_id}")
                e_notes = App.text_input("备注", value=cur.notes or "", key=f"seg_e_notes_{edit_id}")

            if App.button("💾 保存编辑", key="seg_save_edit"):
                # 构建变更记录
                changes = []
                if e_loc != cur.location: changes.append(f"位置: {cur.location} -> {e_loc}")
                if e_dia != cur.diameter: changes.append(f"管径: {cur.diameter} -> {e_dia}")
                if e_len != cur.length: changes.append(f"长度: {cur.length} -> {e_len}")
                
                segment_service.update_segment(
                    edit_id,
                    location=e_loc, material=e_mat,
                    diameter=e_dia or None, length=e_len or None,
                    install_year=e_year or None, notes=e_notes,
                )
                audit_service.log_action(user_info["id"], user_info["username"], "修改", "管段台账", f"修改了管段 {cur.segment_code}。变更内容: {', '.join(changes) if changes else '无核心参数变更'}")
                App.success(f"管段 {cur.segment_code} 已更新")
                App.rerun()

        App.markdown("---")
    elif segments and user_info.get("role") != "admin":
        App.info("💡 只有管理员有权编辑或新增管段台账。")

    # ── 新增管段 ──────────────────────────────────────────────
    if user_info.get("role") == "admin":
        App.markdown("#### ➕ 新增管段")
        nc1, nc2, nc3 = App.columns(3)
        with nc1:
            n_code = App.text_input("管段编号（唯一）", placeholder="如 P-001", key="seg_n_code")
            n_loc  = App.text_input("位置描述", placeholder="如 XX路XX号附近", key="seg_n_loc")
        with nc2:
            n_dia  = App.number_input("管径(mm)", min_value=0.0, value=300.0, step=1.0, key="seg_n_dia")
            n_mat  = App.selectbox("材质", ["混凝土", "PVC", "钢管", "球墨铸铁", "陶瓷", "其他"], key="seg_n_mat")
        with nc3:
            n_len  = App.number_input("长度(m)", min_value=0.0, value=0.0, step=0.1, key="seg_n_len")
            n_year = App.number_input("安装年份", min_value=1900, max_value=2100, value=2010, step=1, key="seg_n_year")
        n_notes = App.text_input("备注", key="seg_n_notes")

        if App.button("➕ 新增管段", key="seg_add_btn"):
            if not n_code.strip():
                App.warning("管段编号不能为空")
            else:
                try:
                    segment_service.create_segment(
                        segment_code=n_code.strip(),
                        location=n_loc, diameter=n_dia or None,
                        material=n_mat, length=n_len or None,
                        install_year=n_year or None, notes=n_notes,
                    )
                    audit_service.log_action(user_info["id"], user_info["username"], "新增", "管段台账", f"新增了管段 {n_code}")
                    App.success(f"管段 {n_code} 已创建")
                    App.rerun()
                except Exception as e:
                    App.error(f"创建失败: {e}")
    else:
        App.info("💡 只有管理员有权新增管段台账。")

    if not segments:
        return

    App.markdown("---")
    App.info("💡 提示：检测任务与管段的关联已优化为在【巡检诊断】或【任务下达】时实时同步，无需在此手动分配。")
