"""
管道检测系统数据库模块
使用SQLAlchemy ORM进行数据库操作
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import os

# 数据库配置
DATABASE_URL = "sqlite:///pipeline_detection.db"
Base = declarative_base()

class User(Base):
    """用户表"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default='user')  # admin, worker, viewer
    full_name = Column(String(50))             # 姓名
    phone = Column(String(20))                 # 电话
    permissions = Column(Text)                 # 权限，JSON字符串格式
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    status = Column(String(20), default='pending')  # active, pending, disabled (用于账号申请审批)
    
    # 关系
    tasks = relationship("DetectionTask", back_populates="user")
    models = relationship("Model", back_populates="user")
    logs = relationship("SystemLog", back_populates="user")

class Model(Base):
    """模型表"""
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    model_type = Column(String(20), nullable=False)  # YOLO, ONNX, TensorRT
    file_path = Column(String(255), nullable=False)
    file_size = Column(Integer)  # 文件大小(字节)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 外键
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # 关系
    user = relationship("User", back_populates="models")
    tasks = relationship("DetectionTask", back_populates="model")

class DetectionTask(Base):
    """检测任务表"""
    __tablename__ = 'detection_tasks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_name = Column(String(100), nullable=False)
    task_type = Column(String(20), nullable=False)  # image, video
    input_file_path = Column(String(255), nullable=False)
    input_file_size = Column(Integer)
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)  # 0-100
    parameters = Column(Text)  # JSON格式存储参数
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 外键
    user_id = Column(Integer, ForeignKey('users.id'))
    model_id = Column(Integer, ForeignKey('models.id'))
    segment_id = Column(Integer, ForeignKey('pipeline_segments.id'), nullable=True)

    # 关系
    user = relationship("User", back_populates="tasks")
    model = relationship("Model", back_populates="tasks")
    results = relationship("DetectionResult", back_populates="task")
    work_orders = relationship("WorkOrder", back_populates="task")
    segment = relationship("PipelineSegment", back_populates="tasks")

class DetectionResult(Base):
    """检测结果表"""
    __tablename__ = 'detection_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_number = Column(Integer)  # 视频帧号，图像为1
    shape = Column(String(20))  # Circle, Ellipse, Undefined
    aspect_ratio = Column(Float)
    orientation = Column(String(50))
    deformation = Column(Float)
    risk_level = Column(String(20)) # 高, 中, 低
    iou_score = Column(Float)  # IoU分数
    confidence = Column(Float)  # 检测置信度
    conclusion = Column(String(20))  # Normal, Deformed
    result_data = Column(Text)  # JSON格式存储详细数据
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 外键
    task_id = Column(Integer, ForeignKey('detection_tasks.id'))
    
    # 关系
    task = relationship("DetectionTask", back_populates="results")
    images = relationship("ResultImage", back_populates="result")
    work_orders = relationship("WorkOrder", primaryjoin="DetectionResult.id==WorkOrder.result_id", back_populates="result")


class PipelineSegment(Base):
    """管段台账表"""
    __tablename__ = 'pipeline_segments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_code = Column(String(50), unique=True, nullable=False)  # 如 P-001
    location = Column(String(200))          # 位置描述
    diameter = Column(Float)               # 管径（mm）
    material = Column(String(50))          # 材质，如 混凝土、PVC、钢管
    length = Column(Float)                 # 管段长度（m）
    install_year = Column(Integer)         # 安装年份
    notes = Column(Text)                   # 备注
    last_inspected_at = Column(DateTime)   # 上次检测时间
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    work_orders = relationship("WorkOrder", back_populates="segment")
    tasks = relationship("DetectionTask", back_populates="segment")


class WorkOrder(Base):
    """工单表"""
    __tablename__ = 'work_orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_code = Column(String(50), unique=True, nullable=False)
    title = Column(String(150), nullable=False)
    risk_level = Column(String(20), default='中')
    priority = Column(String(20), default='中')
    status = Column(String(30), default='待指派')  # 待指派, 进行中, 已完成
    assignee = Column(String(80))
    issue_description = Column(Text)
    action_suggestion = Column(Text)
    processing_notes = Column(Text)
    is_forced = Column(Boolean, default=False)  # 是否强制提交
    created_at = Column(DateTime, default=datetime.utcnow)
    dispatched_at = Column(DateTime)
    processing_at = Column(DateTime)
    repaired_at = Column(DateTime)
    rechecked_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 外键
    task_id = Column(Integer, ForeignKey('detection_tasks.id'))
    result_id = Column(Integer, ForeignKey('detection_results.id'))
    initial_result_id = Column(Integer, ForeignKey('detection_results.id'))
    segment_id = Column(Integer, ForeignKey('pipeline_segments.id'))

    # 关系
    task = relationship("DetectionTask", back_populates="work_orders")
    result = relationship("DetectionResult", foreign_keys=[result_id], back_populates="work_orders")
    initial_result = relationship("DetectionResult", foreign_keys=[initial_result_id], overlaps="result")
    segment = relationship("PipelineSegment", back_populates="work_orders")
    logs = relationship("WorkOrderLog", back_populates="work_order")


class WorkOrderLog(Base):
    """工单操作日志表"""
    __tablename__ = 'work_order_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_code = Column(String(50))           # 记录时的工单编号
    action_type = Column(String(30), nullable=False)  # 创建, 更新
    old_status = Column(String(30))
    new_status = Column(String(30))
    risk_level = Column(String(20))           # 记录时的风险等级
    priority = Column(String(20))             # 记录时的优先级
    is_forced = Column(Boolean, default=False) # 是否为强制提交
    assignee = Column(String(80))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 外键
    work_order_id = Column(Integer, ForeignKey('work_orders.id'), nullable=False)

    # 关系
    work_order = relationship("WorkOrder", back_populates="logs")

class ResultImage(Base):
    """结果图像表"""
    __tablename__ = 'result_images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_type = Column(String(20), nullable=False)  # original, segmented, mask, comparison
    image_data = Column(LargeBinary)  # 图像二进制数据
    image_format = Column(String(10), default='PNG')  # PNG, JPG
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 外键
    result_id = Column(Integer, ForeignKey('detection_results.id'))
    
    # 关系
    result = relationship("DetectionResult", back_populates="images")

class SystemConfig(Base):
    """系统配置表"""
    __tablename__ = 'system_configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(50), unique=True, nullable=False)
    config_value = Column(Text, nullable=False)
    config_type = Column(String(20), default='string')  # string, number, boolean, json
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemLog(Base):
    """系统审计日志表"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(80))
    action = Column(String(50), nullable=False)    # 登录, 退出, 增, 删, 改, 查
    module = Column(String(50))                    # 管段台账, 工单管理, 用户管理
    content = Column(Text)                         # 详细内容，如 "修改了管段 P-001 的长度：100 -> 120"
    ip_address = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User", back_populates="logs")

class TaskStatistics(Base):
    """任务统计表"""
    __tablename__ = 'task_statistics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    failed_tasks = Column(Integer, default=0)
    total_images_processed = Column(Integer, default=0)
    total_videos_processed = Column(Integer, default=0)
    average_processing_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

# 数据库连接和会话管理
class DatabaseManager:
    def __init__(self, database_url=DATABASE_URL):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """获取数据库会话"""
        return self.SessionLocal()
        
    def close_session(self, session):
        """关闭数据库会话"""
        session.close()

# 数据库操作类
class DatabaseOperations:
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    def init_database(self):
        """初始化数据库"""
        self.db_manager.create_tables()
        self._migrate()
        self._create_default_configs()
        self._create_default_users()
        
    def _migrate(self):
        """增量迁移：为旧数据库补充新列/新表"""
        conn = self.db_manager.engine.raw_connection()
        try:
            cur = conn.cursor()
            # work_orders 相关列
            cur.execute("PRAGMA table_info(work_orders)")
            wo_cols = {row[1] for row in cur.fetchall()}
            if "segment_id" not in wo_cols:
                cur.execute("ALTER TABLE work_orders ADD COLUMN segment_id INTEGER REFERENCES pipeline_segments(id)")
            if "result_id" not in wo_cols:
                cur.execute("ALTER TABLE work_orders ADD COLUMN result_id INTEGER REFERENCES detection_results(id)")
            if "initial_result_id" not in wo_cols:
                cur.execute("ALTER TABLE work_orders ADD COLUMN initial_result_id INTEGER REFERENCES detection_results(id)")
            if "is_forced" not in wo_cols:
                cur.execute("ALTER TABLE work_orders ADD COLUMN is_forced BOOLEAN DEFAULT 0")

            # detection_tasks 相关列
            cur.execute("PRAGMA table_info(detection_tasks)")
            dt_cols = {row[1] for row in cur.fetchall()}
            if "segment_id" not in dt_cols:
                cur.execute("ALTER TABLE detection_tasks ADD COLUMN segment_id INTEGER REFERENCES pipeline_segments(id)")
            
            # users 相关列
            cur.execute("PRAGMA table_info(users)")
            user_cols = {row[1] for row in cur.fetchall()}
            if "status" not in user_cols:
                cur.execute("ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active'")
            if "full_name" not in user_cols:
                cur.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR(50)")
            if "phone" not in user_cols:
                cur.execute("ALTER TABLE users ADD COLUMN phone VARCHAR(20)")
            if "permissions" not in user_cols:
                cur.execute("ALTER TABLE users ADD COLUMN permissions TEXT")
            
            conn.commit()
        finally:
            conn.close()

    def _create_default_configs(self):
        """创建默认配置"""
        session = self.db_manager.get_session()
        try:
            # 检查是否已有配置
            existing_configs = session.query(SystemConfig).count()
            if existing_configs > 0:
                return
                
            # 默认配置
            default_configs = [
                SystemConfig(
                    config_key="defect_threshold_crack",
                    config_value="0.15",
                    config_type="number",
                    description="裂缝检测阈值"
                ),
                SystemConfig(
                    config_key="defect_threshold_corrosion",
                    config_value="0.2",
                    config_type="number",
                    description="腐蚀检测阈值"
                ),
                SystemConfig(
                    config_key="defect_threshold_deformation",
                    config_value="0.3",
                    config_type="number",
                    description="变形检测阈值"
                ),
                SystemConfig(
                    config_key="batch_size_default",
                    config_value="5",
                    config_type="number",
                    description="默认批处理大小"
                ),
                SystemConfig(
                    config_key="processing_mode_default",
                    config_value="逐帧",
                    config_type="string",
                    description="默认处理模式"
                ),
                SystemConfig(
                    config_key="frame_spacing_default",
                    config_value="30",
                    config_type="number",
                    description="默认帧间距"
                ),
                SystemConfig(
                    config_key="risk_threshold_high",
                    config_value="0.08",
                    config_type="number",
                    description="高风险变形率阈值"
                ),
                SystemConfig(
                    config_key="risk_threshold_mid",
                    config_value="0.03",
                    config_type="number",
                    description="中风险变形率阈值"
                ),
                SystemConfig(
                    config_key="health_penalty_high",
                    config_value="1.5",
                    config_type="number",
                    description="高风险扣分权重 (每1%)"
                ),
                SystemConfig(
                    config_key="health_penalty_mid",
                    config_value="0.5",
                    config_type="number",
                    description="中风险扣分权重 (每1%)"
                )
            ]
            
            for config in default_configs:
                session.add(config)
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def _create_default_users(self):
        """创建默认用户账号"""
        session = self.db_manager.get_session()
        try:
            # 检查是否已有用户
            user_count = session.query(User).count()
            if user_count > 0:
                return

            # 默认用户
            default_users = [
                User(
                    username="admin",
                    email="admin@example.com",
                    password_hash="123456",  # 简单明文，仅供毕设展示使用
                    role="admin",
                    status="active",
                    full_name="系统管理员",
                    permissions=json.dumps({"detect": True, "repair": True, "ledger": True})
                ),
                User(
                    username="worker",
                    email="worker@example.com",
                    password_hash="123456",
                    role="worker",
                    status="active",
                    full_name="现场工程师",
                    permissions=json.dumps({"detect": True, "repair": True, "ledger": True})
                )
            ]

            for user in default_users:
                session.add(user)
            session.commit()
            print("✅ 默认用户账号创建成功")
        except Exception as e:
            session.rollback()
            print(f"❌ 创建默认用户失败: {e}")
        finally:
            self.db_manager.close_session(session)
    
    def get_config(self, key, default_value=None):
        """获取配置值"""
        session = self.db_manager.get_session()
        try:
            config = session.query(SystemConfig).filter(
                SystemConfig.config_key == key,
                SystemConfig.is_active == True
            ).first()
            
            if config:
                if config.config_type == "number":
                    return float(config.config_value)
                elif config.config_type == "boolean":
                    return config.config_value.lower() == "true"
                elif config.config_type == "json":
                    return json.loads(config.config_value)
                else:
                    return config.config_value
            return default_value
            
        finally:
            self.db_manager.close_session(session)
    
    def set_config(self, key, value, config_type="string", description=""):
        """设置配置值"""
        session = self.db_manager.get_session()
        try:
            config = session.query(SystemConfig).filter(
                SystemConfig.config_key == key
            ).first()
            
            if config:
                config.config_value = str(value)
                config.config_type = config_type
                config.updated_at = datetime.utcnow()
            else:
                config = SystemConfig(
                    config_key=key,
                    config_value=str(value),
                    config_type=config_type,
                    description=description
                )
                session.add(config)
                
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

# 全局数据库实例
db_ops = DatabaseOperations()

