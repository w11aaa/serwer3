"""
数据库服务层
提供高级数据库操作接口
"""

from database import (
    DatabaseManager, DetectionTask, DetectionResult, ResultImage, 
    Model, User, WorkOrder, WorkOrderLog, PipelineSegment, SystemLog,
    SystemConfig
)
from sqlalchemy import desc, func, and_, or_, case
from datetime import datetime, timedelta
import json
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from uuid import uuid4

from sqlalchemy.orm import joinedload

class DetectionService:
    """检测服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def list_results(self, task_id=None, limit=1000):
        """列出检测结果"""
        session = self.db_manager.get_session()
        try:
            query = session.query(DetectionResult).options(
                joinedload(DetectionResult.task).joinedload(DetectionTask.segment)
            )
            if task_id:
                query = query.filter(DetectionResult.task_id == task_id)
            return query.order_by(desc(DetectionResult.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

    def create_task(self, task_name, task_type, input_file_path, user_id, model_id, parameters=None, segment_id=None):
        """创建检测任务"""
        session = self.db_manager.get_session()
        try:
            # 获取文件大小
            import os
            file_size = os.path.getsize(input_file_path) if os.path.exists(input_file_path) else 0
            
            task = DetectionTask(
                task_name=task_name,
                task_type=task_type,
                input_file_path=input_file_path,
                input_file_size=file_size,
                user_id=user_id,
                model_id=model_id,
                segment_id=segment_id,
                parameters=json.dumps(parameters) if parameters else None,
                status='pending'
            )
            
            session.add(task)
            session.commit()
            return task.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def update_task_status(self, task_id, status, progress=None):
        """更新任务状态"""
        session = self.db_manager.get_session()
        try:
            task = session.query(DetectionTask).filter(DetectionTask.id == task_id).first()
            if task:
                task.status = status
                if progress is not None:
                    task.progress = progress
                
                if status == 'processing' and not task.started_at:
                    task.started_at = datetime.utcnow()
                elif status in ['completed', 'failed']:
                    task.completed_at = datetime.utcnow()
                
                session.commit()
                
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def get_task_status(self, task_id):
        """获取任务状态"""
        session = self.db_manager.get_session()
        try:
            task = session.query(DetectionTask).filter(DetectionTask.id == task_id).first()
            if task:
                return {
                    'status': task.status,
                    'progress': task.progress,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at
                }
            return None
        finally:
            self.db_manager.close_session(session)

    def list_tasks(self, user_id=None, task_type=None, status=None, limit=100):
        """列出检测任务"""
        session = self.db_manager.get_session()
        try:
            query = session.query(DetectionTask)
            if user_id:
                query = query.filter(DetectionTask.user_id == user_id)
            if task_type:
                query = query.filter(DetectionTask.task_type == task_type)
            if status:
                query = query.filter(DetectionTask.status == status)
            
            return query.order_by(desc(DetectionTask.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

    def delete_tasks(self, task_ids):
        """批量删除检测任务及其关联结果和图像"""
        if not task_ids:
            return 0
        session = self.db_manager.get_session()
        try:
            # 1. 解除工单关联 (将工单中的 task_id 和 result_id 置为 NULL)
            session.query(WorkOrder).filter(WorkOrder.task_id.in_(task_ids)).update({WorkOrder.task_id: None}, synchronize_session=False)
            
            # 获取所有关联的结果ID
            result_ids = [r.id for r in session.query(DetectionResult.id).filter(DetectionResult.task_id.in_(task_ids)).all()]
            
            if result_ids:
                session.query(WorkOrder).filter(WorkOrder.result_id.in_(result_ids)).update({WorkOrder.result_id: None}, synchronize_session=False)
                # 2. 删除结果图像
                session.query(ResultImage).filter(ResultImage.result_id.in_(result_ids)).delete(synchronize_session=False)
                # 3. 删除检测结果
                session.query(DetectionResult).filter(DetectionResult.id.in_(result_ids)).delete(synchronize_session=False)
            
            # 4. 删除任务
            deleted_count = session.query(DetectionTask).filter(DetectionTask.id.in_(task_ids)).delete(synchronize_session=False)
            
            session.commit()
            return deleted_count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def save_detection_result(self, task_id, frame_number, analysis_data, images_data):
        """保存检测结果"""
        session = self.db_manager.get_session()
        try:
            # 创建检测结果记录
            result = DetectionResult(
                task_id=task_id,
                frame_number=frame_number,
                shape=analysis_data.get('Shape', [None])[0] if isinstance(analysis_data.get('Shape'), list) and analysis_data.get('Shape') else analysis_data.get('Shape'),
                aspect_ratio=analysis_data.get('AspectRatio', [None])[0] if isinstance(analysis_data.get('AspectRatio'), list) and analysis_data.get('AspectRatio') else analysis_data.get('AspectRatio'),
                orientation=analysis_data.get('Orientation', [None])[0] if isinstance(analysis_data.get('Orientation'), list) and analysis_data.get('Orientation') else analysis_data.get('Orientation'),
                deformation=analysis_data.get('Deformation', [None])[0] if isinstance(analysis_data.get('Deformation'), list) and analysis_data.get('Deformation') else analysis_data.get('Deformation'),
                risk_level=analysis_data.get('RiskLevel', [None])[0] if isinstance(analysis_data.get('RiskLevel'), list) and analysis_data.get('RiskLevel') else analysis_data.get('RiskLevel'),
                conclusion=analysis_data.get('Conclusion', [None])[0] if isinstance(analysis_data.get('Conclusion'), list) and analysis_data.get('Conclusion') else analysis_data.get('Conclusion'),
                result_data=json.dumps(analysis_data),
                created_at=datetime.utcnow()
            )
            
            session.add(result)
            session.flush()  # 获取result.id
            
            # 保存图像数据
            for image_type, image_array in images_data.items():
                if image_array is not None:
                    # 转换numpy数组为PIL图像
                    if len(image_array.shape) == 3:
                        if image_array.shape[2] == 3:  # RGB
                            pil_image = Image.fromarray(image_array)
                        elif image_array.shape[2] == 4:  # RGBA
                            pil_image = Image.fromarray(image_array)
                        else:
                            continue
                    elif len(image_array.shape) == 2:  # 灰度图
                        pil_image = Image.fromarray(image_array, mode='L')
                    else:
                        continue
                    
                    # 转换为字节数据
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                    
                    result_image = ResultImage(
                        result_id=result.id,
                        image_type=image_type,
                        image_data=img_data,
                        image_format='PNG',
                        width=image_array.shape[1],
                        height=image_array.shape[0],
                        file_size=len(img_data)
                    )
                    
                    session.add(result_image)
            
            session.commit()
            return result.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_task_results(self, task_id):
        """获取任务结果"""
        session = self.db_manager.get_session()
        try:
            results = session.query(DetectionResult).filter(
                DetectionResult.task_id == task_id
            ).order_by(DetectionResult.frame_number).all()
            
            return results
            
        finally:
            self.db_manager.close_session(session)
    
    def get_result_images(self, result_id):
        """获取结果图像"""
        session = self.db_manager.get_session()
        try:
            images = session.query(ResultImage).filter(
                ResultImage.result_id == result_id
            ).all()
            
            return images
            
        finally:
            self.db_manager.close_session(session)

class ModelService:
    """模型服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def register_model(self, name, version, model_type, file_path, user_id, description=""):
        """注册模型"""
        session = self.db_manager.get_session()
        try:
            import os
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            model = Model(
                name=name,
                version=version,
                model_type=model_type,
                file_path=file_path,
                file_size=file_size,
                description=description,
                user_id=user_id,
                is_active=True
            )
            
            session.add(model)
            session.commit()
            return model.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_active_models(self, user_id=None):
        """获取活跃模型"""
        session = self.db_manager.get_session()
        try:
            query = session.query(Model).filter(Model.is_active == True)
            if user_id:
                query = query.filter(Model.user_id == user_id)
            
            return query.order_by(desc(Model.created_at)).all()
            
        finally:
            self.db_manager.close_session(session)

class StatisticsService:
    """统计服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def get_task_statistics(self, days=30):
        """获取任务统计"""
        session = self.db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 总任务数
            total_tasks = session.query(DetectionTask).filter(
                DetectionTask.created_at >= start_date
            ).count()
            
            # 完成任务数
            completed_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.status == 'completed'
                )
            ).count()
            
            # 失败任务数
            failed_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.status == 'failed'
                )
            ).count()
            
            # 处理的任务类型统计
            image_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.task_type == 'image'
                )
            ).count()
            
            video_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.task_type == 'video'
                )
            ).count()
            
            # 检测结果统计
            total_results = session.query(DetectionResult).filter(
                DetectionResult.created_at >= start_date
            ).count()
            
            # 形状分布统计
            shape_stats = session.query(
                DetectionResult.shape,
                func.count(DetectionResult.id).label('count')
            ).filter(
                DetectionResult.created_at >= start_date
            ).group_by(DetectionResult.shape).all()
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                'image_tasks': image_tasks,
                'video_tasks': video_tasks,
                'total_results': total_results,
                'shape_distribution': {shape: count for shape, count in shape_stats}
            }
            
        finally:
            self.db_manager.close_session(session)
    
    def get_performance_metrics(self, days=7):
        """获取性能指标"""
        session = self.db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 平均处理时间
            completed_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.status == 'completed',
                    DetectionTask.started_at.isnot(None),
                    DetectionTask.completed_at.isnot(None)
                )
            ).all()
            
            if completed_tasks:
                total_time = sum([
                    (task.completed_at - task.started_at).total_seconds()
                    for task in completed_tasks
                ])
                avg_processing_time = total_time / len(completed_tasks)
            else:
                avg_processing_time = 0
            
            # 每日任务统计
            daily_stats = session.query(
                func.date(DetectionTask.created_at).label('date'),
                func.count(DetectionTask.id).label('task_count'),
                func.sum(case((DetectionTask.status == 'completed', 1), else_=0)).label('completed_count')
            ).filter(
                DetectionTask.created_at >= start_date
            ).group_by(func.date(DetectionTask.created_at)).all()
            
            return {
                'average_processing_time': avg_processing_time,
                'daily_statistics': [
                    {
                        'date': stat.date.strftime('%Y-%m-%d') if hasattr(stat.date, 'strftime') else str(stat.date),
                        'total_tasks': stat.task_count,
                        'completed_tasks': stat.completed_count
                    }
                    for stat in daily_stats
                ]
            }
            
        finally:
            self.db_manager.close_session(session)

class UserService:
    """用户服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def create_user(self, username, email, password_hash, role='user', full_name=None, phone=None, permissions=None, status='active'):
        """创建用户"""
        session = self.db_manager.get_session()
        try:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                full_name=full_name,
                phone=phone,
                permissions=json.dumps(permissions) if permissions else None,
                status=status
            )
            
            session.add(user)
            session.commit()
            return user.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def update_user(self, user_id, **kwargs):
        """更新用户信息"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                for key, value in kwargs.items():
                    if hasattr(user, key):
                        if key == 'permissions' and value is not None:
                            setattr(user, key, json.dumps(value))
                        else:
                            setattr(user, key, value)
                session.commit()
                # 显式刷新对象以确保在 session 关闭前加载所有属性
                session.refresh(user)
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating user {user_id}: {str(e)}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_user_by_id(self, user_id):
        """根据ID获取用户"""
        session = self.db_manager.get_session()
        try:
            return session.query(User).filter(User.id == user_id).first()
        finally:
            self.db_manager.close_session(session)

    def get_user_by_username(self, username):
        """根据用户名获取用户"""
        session = self.db_manager.get_session()
        try:
            return session.query(User).filter(User.username == username).first()
        finally:
            self.db_manager.close_session(session)

    def verify_user(self, username, password):
        """验证用户登录 (简单明文验证，支持忽略首尾空格)"""
        if not username or not password:
            return None
        # 统一去除首尾空格，防止输入误差
        clean_username = username.strip()
        clean_password = password.strip()
        
        user = self.get_user_by_username(clean_username)
        # 增加状态检查，只有 active 状态的用户可以登录
        if user and user.password_hash == clean_password:
            if user.status == 'disabled':
                return None
            return user
        return None

    def update_last_login(self, user_id):
        """更新最后登录时间"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def list_users(self, role=None, status=None):
        """列出用户"""
        session = self.db_manager.get_session()
        try:
            query = session.query(User)
            if role:
                query = query.filter(User.role == role)
            if status:
                query = query.filter(User.status == status)
            return query.all()
        finally:
            self.db_manager.close_session(session)

    def update_user_status(self, user_id, status):
        """更新用户状态（审批账号）"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.status = status
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            self.db_manager.close_session(session)

    def delete_user(self, user_id):
        """删除用户"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user and user.role != 'admin': # 禁止删除管理员
                session.delete(user)
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            self.db_manager.close_session(session)

class AuditService:
    """系统审计服务类"""
    def __init__(self):
        self.db_manager = DatabaseManager()

    def log_action(self, user_id, username, action, module, content, ip_address=None):
        """记录审计日志"""
        session = self.db_manager.get_session()
        try:
            log = SystemLog(
                user_id=user_id,
                username=username,
                action=action,
                module=module,
                content=content,
                ip_address=ip_address
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"审计日志记录失败: {e}")
        finally:
            self.db_manager.close_session(session)

    def list_logs(self, user_id=None, limit=500):
        """获取审计日志"""
        session = self.db_manager.get_session()
        try:
            query = session.query(SystemLog)
            if user_id:
                query = query.filter(SystemLog.user_id == user_id)
            return query.order_by(desc(SystemLog.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

class WorkOrderService:
    """工单服务类"""

    STATUS_FLOW = ["待指派", "进行中", "待审核", "已完成"]

    def __init__(self):
        self.db_manager = DatabaseManager()

    def _add_order_log(self, session, work_order_id, action_type, old_status=None, new_status=None, assignee=None, notes=None, order_code=None, risk_level=None, priority=None, is_forced=False):
        log = WorkOrderLog(
            work_order_id=work_order_id,
            order_code=order_code,
            action_type=action_type,
            old_status=old_status,
            new_status=new_status,
            risk_level=risk_level,
            priority=priority,
            is_forced=is_forced,
            assignee=assignee,
            notes=notes,
        )
        session.add(log)

    def _build_order_code(self, session=None):
        for _ in range(8):
            code = f"WO-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{uuid4().hex[:4].upper()}"
            if session is None:
                return code
            exists = session.query(WorkOrder).filter(WorkOrder.order_code == code).first()
            if not exists:
                return code
        return f"WO-{uuid4().hex.upper()}"

    def _risk_to_priority(self, risk_level):
        if risk_level == "高":
            return "高"
        if risk_level == "低":
            return "低"
        return "中"

    def create_work_order(self, title, risk_level="中", task_id=None, result_id=None, segment_id=None, assignee=None, issue_description=None, action_suggestion=None, initial_result_id=None):
        session = self.db_manager.get_session()
        try:
            # 自动根据风险等级设定优先级
            priority = self._risk_to_priority(risk_level)
            
            order = WorkOrder(
                order_code=self._build_order_code(session),
                title=title,
                risk_level=risk_level or "中",
                priority=priority,
                status="待指派",
                assignee=assignee,
                issue_description=issue_description,
                action_suggestion=action_suggestion,
                task_id=task_id,
                result_id=result_id,
                initial_result_id=initial_result_id or result_id, # 如果没有传入，默认当前就是初始结果
                segment_id=segment_id,
            )
            session.add(order)
            session.flush()
            self._add_order_log(
                session,
                work_order_id=order.id,
                order_code=order.order_code,
                risk_level=order.risk_level,
                priority=order.priority,
                is_forced=order.is_forced,
                action_type="创建",
                old_status=None,
                new_status=order.status,
                assignee=order.assignee,
                notes=order.issue_description,
            )
            session.commit()
            return order.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def create_work_orders_from_task(self, task_id, risk_level="高", segment_id=None, assignee=None):
        session = self.db_manager.get_session()
        try:
            results = session.query(DetectionResult).filter(
                DetectionResult.task_id == task_id
            ).order_by(DetectionResult.frame_number).all()

            result_ids = [result.id for result in results]
            existing_result_ids = set()
            if result_ids:
                existing_rows = session.query(WorkOrder.result_id).filter(
                    WorkOrder.result_id.in_(result_ids)
                ).all()
                existing_result_ids = {row[0] for row in existing_rows if row[0] is not None}

            created_count = 0
            for result in results:
                if result.id in existing_result_ids:
                    continue

                parsed = {}
                try:
                    parsed = json.loads(result.result_data) if result.result_data else {}
                except Exception:
                    parsed = {}

                current_risk = parsed.get("RiskLevel")
                action_suggestion = parsed.get("ActionSuggestion")
                if isinstance(current_risk, list):
                    current_risk = current_risk[0] if current_risk else None
                if isinstance(action_suggestion, list):
                    action_suggestion = action_suggestion[0] if action_suggestion else None

                if current_risk != risk_level:
                    continue

                frame_number = result.frame_number if result.frame_number is not None else "未知"
                title = f"任务{task_id}-帧{frame_number}风险处置"
                
                # 如果指定了负责人，状态直接进入"进行中"，否则为"待指派"
                status = "进行中" if assignee else "待指派"
                dispatched_at = datetime.utcnow() if assignee else None
                
                order = WorkOrder(
                    order_code=self._build_order_code(session),
                    title=title,
                    risk_level=current_risk,
                    priority=self._risk_to_priority(current_risk),
                    status=status,
                    issue_description=f"检测结论: {result.conclusion}，形状: {result.shape}，变形: {result.deformation}",
                    action_suggestion=action_suggestion,
                    task_id=task_id,
                    result_id=result.id,
                    initial_result_id=result.id,
                    segment_id=segment_id,
                    assignee=assignee,
                    dispatched_at=dispatched_at
                )
                session.add(order)
                session.flush()
                self._add_order_log(
                    session,
                    work_order_id=order.id,
                    order_code=order.order_code,
                    risk_level=order.risk_level,
                    priority=order.priority,
                    is_forced=order.is_forced,
                    action_type="创建",
                    old_status=None,
                    new_status=order.status,
                    assignee=order.assignee,
                    notes=order.action_suggestion,
                )
                existing_result_ids.add(result.id)
                created_count += 1

            session.commit()
            return created_count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def list_work_orders(self, status=None, is_forced=None, limit=200):
        from sqlalchemy import desc
        from sqlalchemy.orm import joinedload
        session = self.db_manager.get_session()
        try:
            query = session.query(WorkOrder).options(joinedload(WorkOrder.segment))
            if status and status != "全部":
                query = query.filter(WorkOrder.status == status)
            if is_forced is not None:
                query = query.filter(WorkOrder.is_forced == is_forced)
            return query.order_by(desc(WorkOrder.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

    def get_work_order_by_id(self, order_id):
        """通过 ID 获取单个工单"""
        from sqlalchemy.orm import joinedload
        session = self.db_manager.get_session()
        try:
            return session.query(WorkOrder).options(
                joinedload(WorkOrder.segment),
                joinedload(WorkOrder.result)
            ).filter(WorkOrder.id == order_id).first()
        finally:
            self.db_manager.close_session(session)

    def update_work_order(self, order_id, status=None, assignee=None, processing_notes=None, is_forced=None, task_id=None, result_id=None):
        session = self.db_manager.get_session()
        try:
            order = session.query(WorkOrder).filter(WorkOrder.id == order_id).first()
            if not order:
                return False

            old_status = order.status

            if status:
                # 状态流转逻辑优化
                order.status = status
                now = datetime.utcnow()
                if status == "待指派" and order.dispatched_at is None:
                    order.dispatched_at = now
                elif status == "进行中" and order.processing_at is None:
                    order.processing_at = now
                elif status == "已完成":
                    if order.repaired_at is None:
                        order.repaired_at = now
                    order.rechecked_at = now # 闭环即复检
                elif status == "待审核" and order.repaired_at is None:
                    order.repaired_at = now # 提交审核视为完成修复

            if assignee is not None:
                order.assignee = assignee
            if processing_notes is not None:
                # 自动追加时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_note = f"[{timestamp}] {processing_notes}"
                if order.processing_notes:
                    order.processing_notes = f"{order.processing_notes}\n{new_note}"
                else:
                    order.processing_notes = new_note
                    
            if is_forced is not None:
                order.is_forced = is_forced
            if task_id is not None:
                order.task_id = task_id
            if result_id is not None:
                order.result_id = result_id

            self._add_order_log(
                session,
                work_order_id=order.id,
                order_code=order.order_code,
                risk_level=order.risk_level,
                priority=order.priority,
                is_forced=order.is_forced,
                action_type="更新",
                old_status=old_status,
                new_status=order.status,
                assignee=order.assignee,
                notes=processing_notes, # 日志记录原始备注
            )

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def batch_update_work_orders(self, order_ids, status=None, assignee=None, notes=None):
        """批量更新工单"""
        if not order_ids:
            return 0
        
        count = 0
        for oid in order_ids:
            if self.update_work_order(oid, status=status, assignee=assignee, processing_notes=notes):
                count += 1
        return count

    def get_work_order_stats(self, assignee_id=None):
        session = self.db_manager.get_session()
        try:
            from sqlalchemy import func
            query = session.query(WorkOrder)
            
            if assignee_id:
                user = session.query(User).filter(User.id == assignee_id).first()
                if user:
                    query = query.filter(WorkOrder.assignee == user.username)
                else:
                    return {"total": 0, "closed": 0, "closed_rate": 0, "status_distribution": {}}
            
            total = query.count()
            by_status = session.query(
                WorkOrder.status,
                func.count(WorkOrder.id).label('count')
            )
            if assignee_id:
                user = session.query(User).filter(User.id == assignee_id).first()
                if user:
                    by_status = by_status.filter(WorkOrder.assignee == user.username)
            
            by_status = by_status.group_by(WorkOrder.status).all()
            
            closed_query = session.query(WorkOrder).filter(WorkOrder.status == "已完成")
            if assignee_id:
                user = session.query(User).filter(User.id == assignee_id).first()
                if user:
                    closed_query = closed_query.filter(WorkOrder.assignee == user.username)
            
            closed = closed_query.count()
            
            return {
                "total": total,
                "closed": closed,
                "closed_rate": (closed / total * 100) if total > 0 else 0,
                "status_distribution": {status: count for status, count in by_status},
            }
        finally:
            self.db_manager.close_session(session)

    def list_work_order_logs(self, work_order_id, limit=200):
        session = self.db_manager.get_session()
        try:
            return session.query(WorkOrderLog).filter(
                WorkOrderLog.work_order_id == work_order_id
            ).order_by(desc(WorkOrderLog.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

    def delete_work_orders(self, order_ids):
        """批量删除工单及其日志，返回实际删除数量"""
        if not order_ids:
            return 0
        session = self.db_manager.get_session()
        try:
            session.query(WorkOrderLog).filter(WorkOrderLog.work_order_id.in_(order_ids)).delete(synchronize_session=False)
            deleted = session.query(WorkOrder).filter(WorkOrder.id.in_(order_ids)).delete(synchronize_session=False)
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)


class PipelineSegmentService:
    """管段台账服务"""

    def __init__(self):
        self.db_manager = DatabaseManager()

    def list_segments(self):
        session = self.db_manager.get_session()
        try:
            return session.query(PipelineSegment).order_by(PipelineSegment.segment_code).all()
        finally:
            self.db_manager.close_session(session)

    def get_segment(self, segment_id):
        session = self.db_manager.get_session()
        try:
            return session.query(PipelineSegment).filter(PipelineSegment.id == segment_id).first()
        finally:
            self.db_manager.close_session(session)

    def create_segment(self, segment_code, location="", diameter=None, material="", length=None, install_year=None, notes=""):
        session = self.db_manager.get_session()
        try:
            exists = session.query(PipelineSegment).filter(PipelineSegment.segment_code == segment_code).first()
            if exists:
                raise ValueError(f"管段编号 {segment_code} 已存在")
            seg = PipelineSegment(
                segment_code=segment_code,
                location=location,
                diameter=diameter,
                material=material,
                length=length,
                install_year=install_year,
                notes=notes,
            )
            session.add(seg)
            session.commit()
            return seg.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def update_segment(self, segment_id, **kwargs):
        session = self.db_manager.get_session()
        try:
            seg = session.query(PipelineSegment).filter(PipelineSegment.id == segment_id).first()
            if not seg:
                return False
            for k, v in kwargs.items():
                if hasattr(seg, k):
                    setattr(seg, k, v)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def delete_segment(self, segment_id):
        session = self.db_manager.get_session()
        try:
            seg = session.query(PipelineSegment).filter(PipelineSegment.id == segment_id).first()
            if not seg:
                return False
            session.delete(seg)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def update_last_inspected(self, segment_id, dt=None):
        return self.update_segment(segment_id, last_inspected_at=dt or datetime.utcnow())

    def assign_tasks_to_segment(self, task_ids, segment_id):
        """将一批任务分配给指定管段"""
        session = self.db_manager.get_session()
        try:
            session.query(DetectionTask).filter(
                DetectionTask.id.in_(task_ids)
            ).update({DetectionTask.segment_id: segment_id}, synchronize_session=False)
            session.commit()
            return len(task_ids)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def unassign_tasks(self, task_ids):
        """取消任务的管段分配"""
        session = self.db_manager.get_session()
        try:
            session.query(DetectionTask).filter(
                DetectionTask.id.in_(task_ids)
            ).update({DetectionTask.segment_id: None}, synchronize_session=False)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def get_tasks_by_segment(self, segment_id):
        """获取某管段下的所有任务"""
        from sqlalchemy.orm import joinedload
        session = self.db_manager.get_session()
        try:
            return session.query(DetectionTask).filter(
                DetectionTask.segment_id == segment_id
            ).order_by(DetectionTask.id).all()
        finally:
            self.db_manager.close_session(session)

    def get_unassigned_tasks(self):
        """获取未分配管段的任务"""
        session = self.db_manager.get_session()
        try:
            return session.query(DetectionTask).filter(
                DetectionTask.segment_id == None
            ).order_by(DetectionTask.id).all()
        finally:
            self.db_manager.close_session(session)

    def assign_task_id_range(self, segment_id, id_from, id_to):
        """按 ID 范围批量分配任务到管段"""
        session = self.db_manager.get_session()
        try:
            updated = session.query(DetectionTask).filter(
                DetectionTask.id >= id_from,
                DetectionTask.id <= id_to,
            ).update({DetectionTask.segment_id: segment_id}, synchronize_session=False)
            session.commit()
            return updated
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)


class ConfigService:
    """系统配置服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self._cache = {}

    def get_config(self, key, default=None, config_type='string'):
        """获取配置项"""
        if key in self._cache:
            return self._cache[key]
            
        session = self.db_manager.get_session()
        try:
            config = session.query(SystemConfig).filter(SystemConfig.config_key == key).first()
            if config:
                value = config.config_value
                if config.config_type == 'number':
                    try: value = float(value)
                    except: pass
                elif config.config_type == 'boolean':
                    value = value.lower() == 'true'
                elif config.config_type == 'json':
                    try: value = json.loads(value)
                    except: pass
                
                self._cache[key] = value
                return value
            
            # 如果不存在，尝试创建默认值
            if default is not None:
                self.set_config(key, default, config_type)
            return default
        finally:
            self.db_manager.close_session(session)

    def set_config(self, key, value, config_type='string', description=None):
        """设置配置项"""
        session = self.db_manager.get_session()
        try:
            config = session.query(SystemConfig).filter(SystemConfig.config_key == key).first()
            
            val_str = str(value)
            if config_type == 'json':
                val_str = json.dumps(value)
            elif config_type == 'boolean':
                val_str = 'true' if value else 'false'
                
            if config:
                config.config_value = val_str
                config.config_type = config_type
                if description:
                    config.description = description
                config.updated_at = datetime.utcnow()
            else:
                config = SystemConfig(
                    config_key=key,
                    config_value=val_str,
                    config_type=config_type,
                    description=description
                )
                session.add(config)
            
            session.commit()
            self._cache[key] = value # 更新缓存
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def get_all_configs(self):
        """获取所有配置"""
        session = self.db_manager.get_session()
        try:
            configs = session.query(SystemConfig).all()
            res = {}
            for c in configs:
                val = c.config_value
                if c.config_type == 'number':
                    try: val = float(val)
                    except: pass
                elif c.config_type == 'boolean':
                    val = val.lower() == 'true'
                elif c.config_type == 'json':
                    try: val = json.loads(val)
                    except: pass
                res[c.config_key] = {
                    "value": val,
                    "type": c.config_type,
                    "description": c.description
                }
            return res
        finally:
            self.db_manager.close_session(session)

# 全局服务实例
detection_service = DetectionService()
model_service = ModelService()
statistics_service = StatisticsService()
user_service = UserService()
audit_service = AuditService()
workorder_service = WorkOrderService()
segment_service = PipelineSegmentService()
config_service = ConfigService()
