from Analysis import AnalyzePicture, ShowMask, AnalizeMask, GetDictData
from Library import App, Temp, Shutil, OS, CV2, Torch, NP
from Caching import BuildModel
from Export import GetResults, GetImageResults
from Visualize import ShowVisuals
from db_integration import (
    db_integration, show_database_dashboard, show_analysis_history, 
    show_workorder_dashboard, show_settings_dashboard, show_pipeline_ledger,
    show_worker_task_center, show_report_center
)
from db_service import user_service, workorder_service, audit_service, segment_service, detection_service
from TaskQueue import task_worker
from HealthScore import calc_health_score
from Explainability import build_single_explainability, build_ensemble_explainability, render_explainability_panel
import time
from datetime import datetime
import json

# 必须作为第一个 Streamlit 命令调用，防止布局重影和配置冲突
App.set_page_config(
    page_title="城市下水道智能运维平台",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_PROFILES = {
    "YOLO11L-Seg.pt": "高速优先（轻量）",
    "YOLO9C-Seg.pt": "平衡模式（推荐）",
    "YOLO8L-Seg.pt": "精度优先（重型）",
}

DEFAULT_MODEL_ORDER = ["YOLO11L-Seg.pt", "YOLO9C-Seg.pt", "YOLO8L-Seg.pt"]
DEFAULT_NORMAL_MODEL = "YOLO9C-Seg.pt"
DEFAULT_FAST_MODEL = "YOLO11L-Seg.pt"

# ── 核心 UI 样式 ──────────────────────────────────────────────────
def load_css():
    App.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { border-radius: 8px; font-weight: 500; }
        .task-card { 
            background: white; padding: 1.2rem; border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1rem;
        }
        .status-badge {
            padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold;
        }
        h1, h2, h3 { color: #1e293b; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9; border-radius: 8px 8px 0 0; padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if "logged_in" not in App.session_state:
        App.session_state["logged_in"] = False
    if "user_info" not in App.session_state:
        App.session_state["user_info"] = None
    if "WorkDir" not in App.session_state:
        work = OS.path.join(Temp.gettempdir(), "StreamlitSewer")
        if OS.path.exists(work): Shutil.rmtree(work)
        OS.makedirs(work)
        App.session_state["WorkDir"] = work
        App.session_state["MODELS"] = {}
        App.session_state["MODELS_LOADED"] = False
    # 结果缓存
    for key in ["VIDEO_RESULTS", "IMAGE_RESULTS", "IMAGE_HS", "VIDEO_HS", "INSPECTION_RESULTS", "REPAIRED_KEYS", "WORKER_SCAN_RES"]:
        if key not in App.session_state: App.session_state[key] = [] if "RESULTS" in key or "KEYS" in key else None

# ── 全局异常拦截 ──────────────────────────────────────────────────
def global_exception_handler(func):
    """全局异常拦截装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            App.error(f"🚨 系统运行中遇到严重错误，请联系管理员。错误发生在: {func.__name__}")
            
            # 记录到系统日志
            try:
                user_info = App.session_state.get("user_info", {})
                from db_service import audit_service
                audit_service.log_action(
                    user_id=user_info.get("id", 0),
                    username=user_info.get("username", "system"),
                    action="系统崩溃",
                    module="Global",
                    content=f"Exception in {func.__name__}: {str(e)}\n{err_msg[:500]}"
                )
            except: pass

            with App.expander("🔍 错误详细信息 (提供给管理员)", expanded=True):
                App.error(f"异常类型: {type(e).__name__}")
                App.error(f"异常内容: {str(e)}")
                App.code(err_msg, language="python")
            
            if App.button("🔄 尝试重启应用", key="btn_crash_restart"):
                App.rerun()
            App.stop()
    return wrapper

# ── 模型加载 ──────────────────────────────────────────────────────
@global_exception_handler
def preload_models():
    if App.session_state.get("MODELS_LOADED"):
        return App.session_state["MODELS"]
    
    models_dir = OS.path.join(OS.path.dirname(OS.path.dirname(OS.path.abspath(__file__))), "Models")
    loaded = {}
    with App.spinner("🤖 正在启动智能检测引擎..."):
        for name in DEFAULT_MODEL_ORDER:
            # 优先寻找加速格式 (.engine > .onnx > .pt)
            base_name = name.rsplit('.', 1)[0]
            found_path = None
            found_ext = None
            
            for ext in [".engine", ".onnx", ".pt"]:
                p = OS.path.join(models_dir, f"{base_name}{ext}")
                if OS.path.exists(p):
                    found_path = p
                    found_ext = ext
                    break
            
            if found_path:
                try:
                    model = BuildModel(found_path, found_ext, "auto")
                    if model: loaded[name] = model
                except Exception as e:
                    App.warning(f"加载模型 {name} 失败: {str(e)}")
    App.session_state["MODELS"] = loaded
    App.session_state["MODELS_LOADED"] = True
    return loaded

# ── 业务逻辑封装 ──────────────────────────────────────────────────
def _create_inspection_work_order(name, res, segment_id=None):
    """为巡检结果下达维修任务"""
    try:
        # 报修时先保存当前的分析结果到数据库，获取 result_id
        # res["plot"] 是图像数据，res["data"] 是数据字典，res["mask"] 是掩码
        task_id, result_id = db_integration.save_image_analysis(
            None, res["data"], res["plot"], res["mask"], segment_id=segment_id
        )

        workorder_service.create_work_order(
            title=f"巡检发现问题: {name}",
            issue_description=f"AI 自动检出风险：{res['risk']}。诊断结论：{res['data'].get('ActionSuggestion')[0]}",
            risk_level=res["risk"],
            segment_id=segment_id,
            result_id=result_id, # 关联初始报修结果
            task_id=task_id      # 关联初始任务
        )
        if segment_id:
            segment_service.update_last_inspected(segment_id)
        audit_service.log_action(App.session_state["user_info"]["id"], App.session_state["user_info"]["username"], "报修", "巡检", f"为 {name} 创建了维修任务")
        return True
    except Exception as e:
        App.error(f"下达任务失败: {str(e)}")
        return False

def get_mask_from_result(result, target_hw):
    target_h, target_w = target_hw
    if result.masks is None: return NP.zeros((target_h, target_w), dtype=NP.uint8)
    mask = (result.masks.data[0].cpu().numpy() * 255).astype(NP.uint8)
    if mask.shape != (target_h, target_w):
        mask = CV2.resize(mask, (target_w, target_h), interpolation=CV2.INTER_NEAREST)
    return mask

def run_smart_inference(image_input, mode="平衡模式"):
    models = App.session_state.get("MODELS", {})
    if not models:
        models = preload_models()
    if not models:
        App.error("⚠️ 未能加载模型，请检查 Models 文件夹是否存在预训练权重。")
        return None
    
    # 获取动态阈值
    from database import db_ops
    th_high = float(db_ops.get_config("risk_threshold_high"))
    th_mid  = float(db_ops.get_config("risk_threshold_mid"))
    
    # 映射档位到实际逻辑
    if mode == "深度精检":
        # 集成模式
        image_bgr = CV2.imread(image_input) if isinstance(image_input, str) else image_input.copy()
        h, w = image_bgr.shape[:2]
        masks = []
        start = time.time()
        for m in models.values():
            res = m(image_bgr, conf=0.25)[0]
            masks.append(get_mask_from_result(res, (h, w)))
        
        fused = (NP.sum([(m > 127).astype(NP.uint8) for m in masks], axis=0) >= 2).astype(NP.uint8) * 255
        analysis = AnalizeMask(fused)
        if analysis:
            mask_img, ellipse, shape, ar, orient, deform = analysis
            draw_mask = ShowMask(mask_img, ellipse, shape)
            data = GetDictData(shape, ar, orient, deform, risk_threshold_high=th_high, risk_threshold_mid=th_mid)
        else:
            draw_mask, data = fused, GetDictData(None, None, None, None, risk_threshold_high=th_high, risk_threshold_mid=th_mid)
        
        return {
            "plot": CV2.cvtColor(CV2.addWeighted(image_bgr, 1.0, CV2.merge([NP.zeros_like(fused), fused, NP.zeros_like(fused)]), 0.45, 0), CV2.COLOR_BGR2RGB),
            "mask": draw_mask, "data": data, "elapsed": round((time.time()-start)*1000, 1),
            "risk": data.get("RiskLevel")[0] if data.get("RiskLevel") else "低"
        }
    else:
        # 单模型模式
        m_name = "YOLO9C-Seg.pt" if mode == "平衡模式" else "YOLO11L-Seg.pt"
        if m_name not in models: m_name = list(models.keys())[0]
        model = models[m_name]
        
        start = time.time()
        res = model(image_input, conf=0.25)[0]
        (plot, mask, ellipse), data = AnalyzePicture(res, risk_threshold_high=th_high, risk_threshold_mid=th_mid)
        return {
            "plot": plot, "mask": ShowMask(mask, ellipse, data.get("Shape")[0]), 
            "data": data, "elapsed": round((time.time()-start)*1000, 1),
            "risk": data.get("RiskLevel")[0] if data.get("RiskLevel") else "低"
        }

# ── 页面组件 ──────────────────────────────────────────────────────
def _collect_files_from_path(folder_path: str):
    """递归收集指定目录下所有支持的图片/视频文件路径"""
    supported = {".jpg", ".jpeg", ".png", ".mp4", ".avi"}
    collected = []
    if OS.path.isfile(folder_path):
        if OS.path.splitext(folder_path)[1].lower() in supported:
            collected.append(folder_path)
    elif OS.path.isdir(folder_path):
        for root, _, fnames in OS.walk(folder_path):
            for fn in fnames:
                if OS.path.splitext(fn)[1].lower() in supported:
                    collected.append(OS.path.join(root, fn))
    return collected

from TaskQueue import task_worker

def background_video_analysis(task_id, save_path, inspect_mode, segment_id):
    """后台执行视频分析任务"""
    import cv2 as CV2
    from database import db_ops
    import db_integration
    
    try:
        cap = CV2.VideoCapture(save_path)
        total_frames = int(cap.get(CV2.CAP_PROP_FRAME_COUNT))
        v_res = []
        idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 这里的采样率可以从配置读取
            spacing = int(db_ops.get_config("frame_spacing_default", 30))
            if idx % spacing == 0:
                res = run_smart_inference(frame, inspect_mode)
                if res: v_res.append(res)
                
                # 更新进度
                progress = min(99.0, (idx / total_frames) * 100)
                detection_service.update_task_status(task_id, "processing", progress)
                
            idx += 1
        cap.release()
        
        if v_res:
            db_integration.save_video_analysis(
                save_path, 
                [{"Plot": r["plot"], "Mask": r["mask"], "Shape": r["data"]["Shape"][0], "RiskLevel": r["risk"]} for r in v_res], 
                {"mode": inspect_mode}, 
                segment_id=segment_id,
                existing_task_id=task_id # 传入已存在的 task_id
            )
        
        detection_service.update_task_status(task_id, "completed", 100.0)
        
        # 添加审计日志
        try:
            task = detection_service.get_task_status(task_id) # 虽然状态已更新，但我们需要获取任务名
            # 为了获取更全信息，我们可以直接从数据库查一下，或者这里简化记录
            audit_service.log_action(0, "system", "任务完成", "后台分析", f"视频分析任务 (ID: {task_id}) 已顺利完成")
        except: pass

    except Exception as e:
        detection_service.update_task_status(task_id, "failed", 0.0)
        # 记录失败日志
        try:
            audit_service.log_action(0, "system", "任务失败", "后台分析", f"视频分析任务 (ID: {task_id}) 失败: {str(e)}")
        except: pass
        print(f"Video analysis background task failed: {str(e)}")

def _retry_background_task(task):
    """重试失败的后台任务"""
    try:
        import json
        params = json.loads(task.parameters) if task.parameters else {}
        inspect_mode = params.get("mode", "平衡模式")
        
        # 将任务状态重置为 pending
        detection_service.update_task_status(task.id, "pending", 0.0)
        
        # 重新加入任务队列
        if task.task_type == "video":
            task_worker.add_task(background_video_analysis, task.id, task.input_file_path, inspect_mode, task.segment_id)
            App.toast(f"✅ 任务 {task.task_name} 已重新提交", icon="🚀")
        else:
            App.warning("目前仅支持重试视频任务")
            
        # 记录审计日志
        user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
        audit_service.log_action(user_info["id"], user_info["username"], "重试任务", "后台监控", f"重新启动任务 (ID: {task.id}, 名称: {task.task_name})")
        
    except Exception as e:
        App.error(f"重试失败: {str(e)}")

def render_inspection_page():
    App.markdown("### 🔍 巡检上传与诊断")

    upload_tab, path_tab = App.tabs(["📤 文件上传", "📁 路径批量导入"])

    # ── 公共设置 ──────────────────────────────────────────────────
    all_segments = segment_service.list_segments()
    seg_options = {f"{s.segment_code} ({s.location or '未知位置'})": s.id for s in all_segments}
    
    col_set1, col_set2 = App.columns([1, 1])
    with col_set1:
        inspect_mode = App.selectbox(
            "检测精度", ["快速扫描", "平衡模式", "深度精检"], index=1,
            help="深度精检将调用多个模型，耗时较长但最准确"
        )
    with col_set2:
        selected_seg_label = App.selectbox(
            "📍 归属管段 (可选)", 
            options=["-- 稍后在报告中选择 --"] + list(seg_options.keys()),
            help="预先选择管段，系统将自动关联检测任务与该管段"
        )
    
    global_segment_id = seg_options.get(selected_seg_label)

    # ── Tab 1: 原有文件上传 ───────────────────────────────────────
    with upload_tab:
        files = App.file_uploader("上传现场照片或视频", ["JPG", "PNG", "MP4", "AVI"], accept_multiple_files=True)
        run_upload = files and App.button("🚀 开始自动化巡检诊断", type="primary", use_container_width=True, key="btn_upload")

    # ── Tab 2: 路径批量导入 ───────────────────────────────────────
    with path_tab:
        App.caption("输入本地文件夹路径或单个文件路径，支持递归扫描子目录")
        path_input = App.text_area(
            "文件/文件夹路径（每行一个）",
            placeholder="例如：\nD:\\inspection\\2024-04-25\nD:\\inspection\\single.jpg",
            height=120,
            key="batch_path_input"
        )
        col_scan, col_run = App.columns([1, 1])
        with col_scan:
            do_scan = App.button("🔎 扫描文件", use_container_width=True, key="btn_scan")
        with col_run:
            run_path = App.button("🚀 开始批量诊断", type="primary", use_container_width=True, key="btn_path_run")

        if do_scan and path_input.strip():
            scanned = []
            for line in path_input.strip().splitlines():
                line = line.strip()
                if line:
                    found = _collect_files_from_path(line)
                    scanned.extend(found)
            App.session_state["_batch_scanned_paths"] = scanned
            if scanned:
                App.success(f"共扫描到 {len(scanned)} 个文件")
                App.dataframe(
                    [{"文件路径": p, "类型": "视频" if p.lower().endswith((".mp4", ".avi")) else "图片"} for p in scanned],
                    use_container_width=True, hide_index=True
                )
            else:
                App.warning("未找到支持的文件，请检查路径是否正确")

    # ── 处理逻辑（上传模式）──────────────────────────────────────
    if run_upload and files:
        # 记录审计日志
        user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
        audit_service.log_action(user_info["id"], user_info["username"], "开始巡检", "巡检诊断", f"通过上传方式开始巡检，文件数: {len(files)}, 模式: {inspect_mode}")
        
        new_results = []
        for f in files:
            is_video = f.name.lower().endswith(('.mp4', '.avi'))
            save_path = OS.path.join(App.session_state["WorkDir"], f.name)
            with open(save_path, "wb") as out: out.write(f.read())
            
            if is_video:
                # 创建后台任务
                model_name = "YOLO9C-Seg.pt" if inspect_mode == "平衡模式" else "YOLO11L-Seg.pt"
                task_id = detection_service.create_task(
                    task_name=f"视频巡检: {f.name}",
                    task_type="video",
                    input_file_path=save_path,
                    user_id=App.session_state["user_info"]["id"],
                    model_id=model_name,
                    parameters={"mode": inspect_mode},
                    segment_id=global_segment_id
                )
                
                # 加入任务队列
                task_worker.add_task(background_video_analysis, task_id, save_path, inspect_mode, global_segment_id)
                App.success(f"✅ 视频 {f.name} 已提交至后台处理，请前往【任务中心】查看进度。")
            else:
                res = run_smart_inference(save_path, inspect_mode)
                if res:
                    db_integration.save_image_analysis(save_path, res["data"], res["plot"], res["mask"], segment_id=global_segment_id)
                    new_results.append({"name": f.name, "res": res})
        
        if new_results:
            App.session_state["INSPECTION_RESULTS"] = new_results
            App.session_state["REPAIRED_KEYS"] = []
            App.toast("✅ 诊断完成！请向下滑动查看详细报告并进行报修。", icon="📋")

    # ── 处理逻辑（路径批量模式）──────────────────────────────────
    if run_path:
        batch_paths = App.session_state.get("_batch_scanned_paths")
        if not batch_paths:
            if path_input.strip():
                batch_paths = []
                for line in path_input.strip().splitlines():
                    line = line.strip()
                    if line: batch_paths.extend(_collect_files_from_path(line))
        
        if not batch_paths:
            App.warning("请先输入路径并点击【扫描文件】，或直接输入路径后点击批量诊断")
        else:
            # 记录审计日志
            user_info = App.session_state.get("user_info", {"id": 0, "username": "system"})
            audit_service.log_action(user_info["id"], user_info["username"], "开始批量巡检", "巡检诊断", f"通过路径批量导入开始巡检，文件数: {len(batch_paths)}, 模式: {inspect_mode}")
            
            App.info(f"开始批量诊断，共 {len(batch_paths)} 个文件...")
            progress = App.progress(0, text="准备中...")
            total = len(batch_paths)
            new_results = []
            for i, fpath in enumerate(batch_paths):
                fname = OS.path.basename(fpath)
                progress.progress((i + 1) / total, text=f"正在处理 ({i+1}/{total}): {fname}")
                is_video = fpath.lower().endswith((".mp4", ".avi"))
                if is_video:
                    # 创建后台任务
                    model_name = "YOLO9C-Seg.pt" if inspect_mode == "平衡模式" else "YOLO11L-Seg.pt"
                    task_id = detection_service.create_task(
                        task_name=f"批量导入: {fname}",
                        task_type="video",
                        input_file_path=fpath,
                        user_id=App.session_state["user_info"]["id"],
                        model_id=model_name,
                        parameters={"mode": inspect_mode},
                        segment_id=global_segment_id
                    )
                    
                    # 加入任务队列
                    task_worker.add_task(background_video_analysis, task_id, fpath, inspect_mode, global_segment_id)
                    App.info(f"已提交视频任务: {fname}")
                else:
                    res = run_smart_inference(fpath, inspect_mode)
                    if res:
                        db_integration.save_image_analysis(fpath, res["data"], res["plot"], res["mask"], segment_id=global_segment_id)
                        new_results.append({"name": fname, "res": res})
            
            progress.empty()
            App.success(f"🎉 批量诊断完成！检出 {len(new_results)} 个风险项。")
            if new_results:
                App.session_state["INSPECTION_RESULTS"] = new_results
                App.session_state["REPAIRED_KEYS"] = []
                App.toast("✅ 批量诊断已就绪，请在下方报告中确认风险点。", icon="🚀")
            App.session_state.pop("_batch_scanned_paths", None)

    # ── 显示诊断结果 ─────────────────────────────────────────────
    _render_inspection_results()

def _render_inspection_results():
    """渲染存储在 session_state 中的诊断结果"""
    results = App.session_state.get("INSPECTION_RESULTS", [])
    if not results: return

    App.markdown("---")
    App.subheader("📋 诊断报告与处理")

    # ── 关联管段选择 ──────────────────────────────────────────────
    all_segments = segment_service.list_segments()
    seg_options = {f"{s.segment_code} ({s.location or '未知位置'})": s.id for s in all_segments}
    
    col_seg1, col_seg2 = App.columns([3, 2])
    with col_seg1:
        selected_seg_label = App.selectbox(
            "📍 关联管段 (必选)", 
            options=["-- 请选择受损管段 --"] + list(seg_options.keys()),
            help="维修任务将自动关联到该管段"
        )
    
    selected_segment_id = seg_options.get(selected_seg_label)
    if not selected_segment_id:
        App.warning("⚠️ 请先选择关联管段，否则无法进行报修操作")

    # ── 报告导出区域 ──────────────────────────────────────────────
    with App.expander("📄 生成规范化诊断报告", expanded=False):
        col_meta1, col_meta2 = App.columns(2)
        with col_meta1:
            report_user = App.text_input("报告人", value=App.session_state.get("user_info", {}).get("username", "管理员"))
            project_name = App.text_input("项目名称", value="下水道截面几何变形检测报告")
        with col_meta2:
            report_date = App.date_input("报告日期")
            # 自动填入选中的管段编号
            default_seg_code = selected_seg_label.split(" (")[0] if selected_segment_id else "S-001"
            segment_code = App.text_input("报告展示编号", value=default_seg_code)
        
        notes = App.text_area("备注信息", placeholder="输入补充说明...")
        
        export_col1, export_col2, export_col3 = App.columns(3)
        
        # 准备导出数据格式
        export_data = []
        for r in results:
            d = r["res"]
            export_data.append({
                "Name": r["name"],
                "Plot": d["plot"],
                "Draw": d["mask"],
                "Shape": d["data"]["Shape"][0],
                "AspectRatio": d["data"]["AspectRatio"][0],
                "Orientation": d["data"]["Orientation"][0],
                "Deformation": d["data"]["Deformation"][0],
                "RiskLevel": d["risk"],
                "ActionSuggestion": d["data"].get("ActionSuggestion", ["无"])[0]
            })

        meta = {
            "operator": report_user,
            "project": project_name,
            "date": str(report_date),
            "segment": segment_code,
            "notes": notes,
            "report_type": "综合巡检报告",
            "conclusion": "警告" if any(x["RiskLevel"] == "高" for x in export_data) else "优良"
        }

        with export_col1:
            if App.button("📊 导出 Excel 报告", use_container_width=True):
                from Export import GetImageExcel
                with App.spinner("正在生成 Excel..."):
                    excel_buf = GetImageExcel(export_data)
                    App.download_button("💾 点击下载 Excel", excel_buf, file_name=f"巡检报告_{segment_code}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        with export_col2:
            if App.button("📕 导出 PDF 报告", use_container_width=True):
                from Export import GetImagePDF
                with App.spinner("正在生成 PDF..."):
                    pdf_buf = GetImagePDF(export_data, meta=meta)
                    App.download_button("💾 点击下载 PDF", pdf_buf, file_name=f"巡检报告_{segment_code}.pdf", mime="application/pdf")
        
        with export_col3:
            if App.button("🎬 导出视频报告", use_container_width=True):
                from Export import GetVideo
                with App.spinner("正在合成视频报告..."):
                    video_data = []
                    for x in export_data:
                        video_data.append({
                            "Plot": x["Plot"],
                            "Draw": x["Draw"],
                            "Shape": [x["Shape"]],
                            "RiskLevel": x["RiskLevel"]
                        })
                    video_buf = GetVideo(video_data)
                    App.download_button("💾 点击下载 MP4", video_buf, file_name=f"视频报告_{segment_code}.mp4", mime="video/mp4")

    # ── 诊断明细展示 ──────────────────────────────────────────────
    head_col1, head_col2, head_col3 = App.columns([2, 2, 1])
    with head_col1:
        App.markdown(f"### 📋 诊断明细")
    
    with head_col2:
        risk_filter = App.multiselect(
            "🔍 筛选风险等级",
            options=["高", "中", "低"],
            default=["高", "中"],
            help="选择要显示的风险等级，默认为您筛选出高/中风险项"
        )
    
    # 过滤显示的结果
    filtered_results = [r for r in results if r["res"]["risk"] in risk_filter]
    App.caption(f"当前显示 {len(filtered_results)} / {len(results)} 项")

    # 批量报修功能：支持自定义勾选要派修的项目
    needs_repair = [r for r in filtered_results if r["res"]["risk"] in ["高", "中"] and r["name"] not in App.session_state["REPAIRED_KEYS"]]
    
    # 初始化批量选择 session
    if "batch_repair_keys" not in App.session_state:
        App.session_state["batch_repair_keys"] = []
    
    if needs_repair:
        with head_col3:
            App.markdown("<br>", unsafe_allow_html=True)
            # 全选/取消全选切换
            all_selected = all(r["name"] in App.session_state["batch_repair_keys"] for r in needs_repair)
            select_all_label = "⭕ 取消全选" if all_selected else "☑️ 全选本页"
            if App.button(select_all_label, key="toggle_select_all", use_container_width=True):
                if all_selected:
                    App.session_state["batch_repair_keys"] = [k for k in App.session_state["batch_repair_keys"] if k not in [r["name"] for r in needs_repair]]
                else:
                    for r in needs_repair:
                        if r["name"] not in App.session_state["batch_repair_keys"]:
                            App.session_state["batch_repair_keys"].append(r["name"])
                App.rerun()
            
            # 已勾选数量提示
            selected_count = len([k for k in App.session_state["batch_repair_keys"] if k in [r["name"] for r in needs_repair]])
            if selected_count > 0:
                App.caption(f"已选 {selected_count} 项待派修")
            
            # 批量下达按钮
            if App.button("🛠️ 批量下达维修任务", type="primary", use_container_width=True, 
                          help=f"为已勾选的 {selected_count} 个项目创建维修任务", 
                          disabled=not selected_segment_id or selected_count == 0):
                success_count = 0
                for item in needs_repair:
                    if item["name"] in App.session_state["batch_repair_keys"]:
                        if _create_inspection_work_order(item["name"], item["res"], segment_id=selected_segment_id):
                            App.session_state["REPAIRED_KEYS"].append(item["name"])
                            App.session_state["batch_repair_keys"].remove(item["name"])
                            success_count += 1
                if success_count > 0:
                    App.toast(f"✅ 成功批量创建 {success_count} 个维修任务！")
                    App.rerun()

    for i, item in enumerate(filtered_results):
        name = item["name"]
        res = item["res"]
        is_repaired = name in App.session_state["REPAIRED_KEYS"]
        is_checked = name in App.session_state.get("batch_repair_keys", [])
        
        with App.container(border=True):
            sel_col, img_col, data_col, act_col = App.columns([1, 2, 2, 1])
            with sel_col:
                if not is_repaired and res["risk"] in ["高", "中"]:
                    if App.checkbox("☑️", value=is_checked, key=f"cb_{i}_{name}"):
                        if name not in App.session_state["batch_repair_keys"]:
                            App.session_state["batch_repair_keys"].append(name)
                            App.rerun()
                elif is_repaired:
                    App.success("✅")
                else:
                    App.info("🟢")
            img_col.image(res["plot"], caption=f"诊断图: {name}")
            data_col.dataframe(res["data"], hide_index=True)
            with act_col:
                App.markdown(f"**{res['risk']}**")
                if res["risk"] in ["高", "中"]:
                    if is_repaired:
                        App.success("✅ 已下达维修")
                    else:
                        App.warning("待下达")
                else:
                    App.info("🟢 状态良好")

def login_page():
    """渲染登录页面"""
    # 使用 empty 容器包装整个登录页，便于彻底清除渲染残留
    login_container = App.empty()
    
    with login_container.container():
        # 清除侧边栏，防止登录页出现侧边栏残留
        App.markdown("""
            <style>
                [data-testid="stSidebar"] { display: none !important; }
                .stMain { margin-left: 0 !important; }
                header { visibility: hidden; }
                footer { visibility: hidden; }
            </style>
        """, unsafe_allow_html=True)

        # 居中显示登录框
        _, col2, _ = App.columns([1, 2, 1])
        
        with col2:
            App.markdown("<div style='text-align:center;margin-top:50px;'><h1>🏙️ 城市下水道智能运维平台</h1><p>请登录以访问您的工作台</p></div>", unsafe_allow_html=True)
            
            with App.container(border=True):
                with App.form("login_form", clear_on_submit=False):
                    u = App.text_input("账号", key="login_user_input", placeholder="请输入用户名")
                    p = App.text_input("密码", type="password", key="login_pwd_input", placeholder="请输入密码")
                    
                    if App.form_submit_button("🚀 进入系统", use_container_width=True):
                        if not u or not p:
                            App.error("请输入账号和密码")
                        else:
                            user = user_service.verify_user(u, p)
                            if user:
                                if user.status == 'pending':
                                    App.warning("您的账号正在审核中，请联系管理员")
                                elif user.status == 'disabled':
                                    App.error("您的账号已被禁用，请联系管理员")
                                else:
                                    # 更新最后登录时间
                                    user_service.update_user(user.id, last_login=datetime.now())
                                    App.session_state.update({
                                        "logged_in": True, 
                                        "user_info": {
                                            "id": user.id, 
                                            "username": user.username, 
                                            "role": user.role,
                                            "full_name": user.full_name
                                        }
                                    })
                                    # 登录成功后彻底清除登录容器
                                    login_container.empty()
                                    App.rerun()
                            else:
                                App.error("账号或密码错误")
            
            App.info("💡 提示：管理员账号 admin / 123456，工人账号 worker / 123456")

def show_task_monitor():
    """管理员端：后台任务监控中心"""
    App.markdown("### 🚀 后台任务监控")
    
    # 顶部操作与统计
    tasks = detection_service.list_tasks(limit=50)
    
    col_stat1, col_stat2 = App.columns([3, 1])
    with col_stat2:
        if App.button("🔄 刷新列表", use_container_width=True):
            App.rerun()
            
    if not tasks:
        App.info("✨ 当前暂无后台处理任务。")
        return

    # 统计指标
    total = len(tasks)
    processing = len([t for t in tasks if t.status == 'processing'])
    completed = len([t for t in tasks if t.status == 'completed'])
    failed = len([t for t in tasks if t.status == 'failed'])
    
    m1, m2, m3, m4 = App.columns(4)
    m1.metric("总任务量", total)
    m2.metric("正在处理", processing)
    m3.metric("已完成", completed)
    m4.metric("执行失败", failed, delta_color="inverse" if failed > 0 else "normal")
    
    App.markdown("---")
    
    # 任务列表展示
    for t in tasks:
        with App.container(border=True):
            c_info, c_status, c_action = App.columns([3, 2, 1])
            
            with c_info:
                icon = "📹" if t.task_type == "video" else "🖼️"
                App.markdown(f"**{icon} {t.task_name}**")
                App.caption(f"📁 路径: {t.input_file_path}")
                if t.created_at:
                    App.caption(f"📅 创建于: {t.created_at.strftime('%m-%d %H:%M:%S')}")
            
            with c_status:
                status_styles = {
                    "pending": ("🟡 等待中", "#64748b"),
                    "processing": ("🔵 正在分析", "#3b82f6"),
                    "completed": ("✅ 分析完成", "#22c55e"),
                    "failed": ("❌ 执行失败", "#ef4444")
                }
                label, color = status_styles.get(t.status, ("未知", "#94a3b8"))
                
                App.markdown(f"<span style='color:{color}; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
                
                if t.status == "processing":
                    prog = t.progress if t.progress else 0.0
                    App.progress(prog / 100.0)
                    App.caption(f"进度: {prog:.1f}%")
                elif t.status == "completed":
                    App.progress(1.0)
                    if t.completed_at and t.started_at:
                        dur = (t.completed_at - t.started_at).total_seconds()
                        App.caption(f"耗时: {dur:.1f}s")
                elif t.status == "failed":
                    App.progress(0.0)
                    App.caption("错误: 资源占用或文件损坏")
            
            with c_action:
                if t.status == "completed":
                    if App.button("📂 查看结果", key=f"view_{t.id}", use_container_width=True):
                        App.session_state["active_tab"] = "📂 历史档案"
                        App.session_state["history_selected_task_id"] = t.id
                        App.rerun()
                elif t.status == "failed":
                    if App.button("🔄 重试", key=f"retry_{t.id}", use_container_width=True):
                        _retry_background_task(t)
                        App.rerun()
                elif t.status == "processing":
                    App.button("⌛ 稍后", key=f"wait_{t.id}", disabled=True, use_container_width=True)

def render_user_management_page():
    """管理员端：工人账号管理页面"""
    App.markdown("### ⚙️ 工人账号管理")
    
    # 顶部统计
    users = user_service.list_users(role='worker')
    App.caption(f"当前共有 {len(users)} 名工人账号")
    
    # 操作栏
    col_act1, col_act2 = App.columns([1, 4])
    with col_act1:
        if App.button("➕ 新增工人账号", type="primary", use_container_width=True):
            App.session_state["show_user_form"] = "add"
    
    # 新增/编辑表单
    form_mode = App.session_state.get("show_user_form")
    if form_mode:
        with App.container(border=True):
            App.markdown(f"#### {'新增' if form_mode == 'add' else '编辑'}账号信息")
            
            # 如果是编辑模式，获取最新的用户数据
            edit_user = None
            if form_mode == "edit":
                user_id = App.session_state.get("edit_user_id")
                if user_id:
                    edit_user = user_service.get_user_by_id(user_id)
                
                if not edit_user:
                    App.error("未找到用户信息，可能已被删除")
                    App.session_state.pop("show_user_form", None)
                    App.rerun()
            
            with App.form("user_form"):
                f1, f2 = App.columns(2)
                u_name = f1.text_input("用户名 (用于登录)", value=edit_user.username if edit_user else "")
                u_pwd = f2.text_input("登录密码", type="password", placeholder="留空则不修改" if edit_user else "必填")
                
                f3, f4 = App.columns(2)
                u_full = f3.text_input("工人姓名", value=edit_user.full_name if edit_user else "")
                u_phone = f4.text_input("联系电话", value=edit_user.phone if edit_user else "")
                
                u_email = App.text_input("电子邮箱", value=edit_user.email if edit_user else f"{u_name}@sewer.com" if u_name else "")
                
                App.markdown("**权限控制**")
                p1, p2, p3 = App.columns(3)
                perm_detect = p1.checkbox("巡检权限", value=True)
                perm_repair = p2.checkbox("维修权限", value=True)
                perm_ledger = p3.checkbox("查看台账", value=True)
                
                submit_col1, submit_col2 = App.columns([1, 5])
                if submit_col1.form_submit_button("保存"):
                    perms = {"detect": perm_detect, "repair": perm_repair, "ledger": perm_ledger}
                    if form_mode == "add":
                        if not u_name or not u_pwd:
                            App.error("用户名和密码必填")
                        else:
                            try:
                                user_service.create_user(u_name, u_email, u_pwd, role='worker', full_name=u_full, phone=u_phone, permissions=perms, status='active')
                                App.success("✅ 账号创建成功")
                                App.session_state.pop("show_user_form")
                                App.rerun()
                            except Exception as e: App.error(f"创建失败: {str(e)}")
                    else:
                        update_data = {"username": u_name, "email": u_email, "full_name": u_full, "phone": u_phone, "permissions": perms}
                        if u_pwd: update_data["password_hash"] = u_pwd
                        if user_service.update_user(edit_user.id, **update_data):
                            App.success("✅ 账号更新成功")
                            App.session_state.pop("show_user_form")
                            App.rerun()
                if submit_col2.form_submit_button("取消"):
                    App.session_state.pop("show_user_form")
                    App.rerun()

    # 用户列表
    if users:
        for u in users:
            with App.container(border=True):
                c1, c2, c3, c4 = App.columns([1, 2, 2, 1])
                c1.markdown(f"**{u.full_name or u.username}**")
                c2.caption(f"📞 {u.phone or '未填写'}")
                
                perms = json.loads(u.permissions) if u.permissions else {}
                p_text = []
                if perms.get("detect"): p_text.append("巡检")
                if perms.get("repair"): p_text.append("维修")
                if perms.get("ledger"): p_text.append("台账")
                c3.caption(f"权限: {'|'.join(p_text) if p_text else '无'}")
                
                with c4:
                    edit_col, del_col = App.columns(2)
                    if edit_col.button("📝", key=f"edit_{u.id}", help="编辑"):
                        App.session_state["show_user_form"] = "edit"
                        App.session_state["edit_user_id"] = u.id
                        App.rerun()
                    if del_col.button("🗑️", key=f"del_{u.id}", help="删除"):
                        if user_service.delete_user(u.id):
                            App.success("已删除")
                            App.rerun()
    else:
        App.info("暂无工人账号，请点击上方按钮添加。")

@global_exception_handler
def run_app():
    init_session_state()
    load_css()

    if not App.session_state["logged_in"]:
        login_page()
        App.stop()

    user = App.session_state["user_info"]
    
    # 记录登录审计（如果是新会话）
    if not App.session_state.get("login_logged"):
        audit_service.log_action(user["id"], user["username"], "登录", "系统", "用户成功登录系统")
        App.session_state["login_logged"] = True

    with App.sidebar:
        App.markdown(f"### 👤 {user['username']}")
        role_display = "管理员" if user['role'] == 'admin' else "作业人员"
        App.caption(f"您好，{role_display}")
        if App.button("退出登录", use_container_width=True):
            audit_service.log_action(user["id"], user["username"], "登出", "系统", "用户退出登录")
            App.session_state["logged_in"] = False
            App.session_state["login_logged"] = False
            App.rerun()
        App.markdown("---")
        App.markdown("#### 📅 今日概况")
        App.info(f"日期: {datetime.now().strftime('%Y-%m-%d')}")

    # 根据角色渲染
    if user["role"] == "admin":
        ADMIN_TABS = ["📊 任务管理", "🔍 巡检检测", "📂 档案台账", "⚙️ 系统管理"]
        
        # 初始化导航状态
        if "admin_tab_radio" not in App.session_state:
            App.session_state["admin_tab_radio"] = ADMIN_TABS[0]
        
        # 处理外部跳转请求
        _jump = App.session_state.pop("active_tab", None)
        if _jump:
            tab_mapping = {
                "🛰️ 调度中心": "📊 任务管理", "📊 报表中心": "📊 任务管理", "📊 任务管理": "📊 任务管理",
                "🚀 任务监控": "🔍 巡检检测", "📂 历史档案": "📂 档案台账", "🗂️ 管网台账": "📂 档案台账"
            }
            target = tab_mapping.get(_jump, _jump)
            if target in ADMIN_TABS:
                App.session_state["admin_tab_radio"] = target

        selected_tab = App.radio(
            "功能导航", ADMIN_TABS,
            horizontal=True,
            label_visibility="collapsed",
            key="admin_tab_radio",
        )
        App.session_state["admin_active_tab"] = selected_tab 
        App.markdown("---")

        @global_exception_handler
        def render_tab(tab):
            if tab == "📊 任务管理":
                # 统一的指挥调度中心与任务台账
                show_workorder_dashboard()
                with App.expander("📊 报表下载"):
                    show_report_center()
            elif tab == "🔍 巡检检测": 
                render_inspection_page()
                with App.expander("🚀 实时任务监控"):
                    show_task_monitor()
            elif tab == "📂 档案台账": 
                show_pipeline_ledger()
                # 如果是从任务监控跳转过来的，默认展开历史记录
                is_history_jump = "history_selected_task_id" in App.session_state
                with App.expander("📜 历史分析记录", expanded=is_history_jump):
                    show_analysis_history()
            elif tab == "⚙️ 系统管理": 
                show_settings_dashboard()
        
        render_tab(selected_tab)
    else:
        WORKER_TABS = ["🛠️ 我的工作台", "📸 拍照验收", "🗺️ 管网查询"]
        if "worker_tab_radio" not in App.session_state:
            App.session_state["worker_tab_radio"] = WORKER_TABS[0]
        
        # 处理外部跳转请求
        _jump = App.session_state.pop("active_tab", None)
        if _jump:
            tab_mapping = { "📋 任务中心": "🛠️ 我的工作台", "👤 个人中心": "🛠️ 我的工作台" }
            target = tab_mapping.get(_jump, _jump)
            if target in WORKER_TABS:
                App.session_state["worker_tab_radio"] = target

        selected_tab = App.radio(
            "功能导航", WORKER_TABS,
            horizontal=True,
            label_visibility="collapsed",
            key="worker_tab_radio",
        )
        App.session_state["worker_active_tab"] = selected_tab
        App.markdown("---")

        @global_exception_handler
        def render_worker_tab(tab):
            if tab == "🛠️ 我的工作台":
                with App.expander("👤 我的个人资料与统计"):
                    show_worker_self_service()
                App.markdown("---")
                show_worker_task_center(user["id"])
            elif tab == "📸 拍照验收": 
                render_worker_acceptance_page()
            elif tab == "🗺️ 管网查询": 
                show_pipeline_ledger()
        
        render_worker_tab(selected_tab)

@global_exception_handler
def render_worker_acceptance_page():
    """工人端现场验收页面"""
    App.markdown("### 📸 现场维修验收")
    
    active_wo = App.session_state.get("active_wo_for_detect")
    if not active_wo:
        App.info("💡 请先在【我的工作台】选择一个要验收的任务，点击【拍照验收】按钮。")
        if App.button("⬅️ 返回工作台", use_container_width=True):
            App.session_state["active_tab"] = "🛠️ 我的工作台"
            App.rerun()
        return

    # 安全获取任务信息
    wo_code = active_wo.get('code', '未知')
    wo_segment = active_wo.get('segment', '未知')
    wo_id = active_wo.get('id')
    
    if not wo_id:
        App.error("任务数据异常，请重新选择")
        return

    App.success(f"正在验收任务: **{wo_code}** (管段: {wo_segment})")
    
    # AI 检测精度选择
    inspect_mode = App.selectbox(
        "AI 检测精度", ["快速扫描", "平衡模式", "深度精检"], index=1,
        help="深度精检将调用多个模型，耗时较长但最准确"
    )
    
    f = App.file_uploader("拍摄或上传维修后的照片", ["JPG", "PNG"])

    if f and App.button("🔬 开始 AI 验收分析", type="primary", use_container_width=True):
        save_path = OS.path.join(App.session_state["WorkDir"], f"verify_{f.name}")
        with open(save_path, "wb") as out: out.write(f.read())
        
        res = run_smart_inference(save_path, inspect_mode)
        if not isinstance(res, dict) or res.get("plot") is None:
            App.error("AI 验收分析失败：模型未就绪或图像未识别成功，请稍后重试。")
        else:
            data = res.get("data") or {}
            deform_list = data.get("Deformation") or [None]
            risk_level = res.get("risk", "未知")
            
            # 获取健康评分
            from HealthScore import calc_health_score
            mock_res = [{"Deformation": deform_list[0], "RiskLevel": risk_level}]
            hs = calc_health_score(mock_res, 0.3, 0.15)
            
            # 保存到 session_state 以便跨 rerun 持久化
            App.session_state["last_analysis_res"] = res
            App.session_state["last_analysis_hs"] = hs
            App.rerun()

    if App.session_state.get("last_analysis_res"):
        res = App.session_state["last_analysis_res"]
        hs = App.session_state["last_analysis_hs"]
        
        with App.container(border=True):
            c1, c2 = App.columns([1, 1])
            c1.image(res["plot"], caption="现场维修后状态")
            with c2:
                App.metric("健康得分", f"{hs['score']}/100")
                App.markdown(f"**诊断结论:** {hs['level']}")
                
                if hs["score"] >= 80:
                    App.success("🎉 AI 评估合格！修得不错。")
                    if App.button("✅ 提交验收并完成任务", type="primary", use_container_width=True):
                        # 保存验收照片到 DetectionResult 和 ResultImage
                        try:
                            images_data = {"segmented": res["plot"]}
                            # 如果没有 task_id，可以传 None 或者 0
                            task_id = active_wo.get("task_id") or 0
                            result_id = detection_service.save_detection_result(
                                task_id=task_id,
                                frame_number=1,
                                analysis_data=res.get("data", {}),
                                images_data=images_data
                            )
                            
                            if workorder_service.update_work_order(
                                 wo_id, status="已完成", 
                                 result_id=result_id,
                                 processing_notes=f"现场验收通过。AI 得分: {hs['score']} ({hs['level']})。"
                             ):
                                audit_service.log_action(
                                    App.session_state["user_info"]["id"], 
                                    App.session_state["user_info"]["username"], 
                                    "验收提交", "现场验收", f"任务 {wo_code} 验收合格并提交"
                                )
                                # 清除状态并通过 active_tab 中转跳转
                                del App.session_state["active_wo_for_detect"]
                                if "last_analysis_res" in App.session_state: del App.session_state["last_analysis_res"]
                                App.session_state["active_tab"] = "🛠️ 我的工作台"
                                App.success("✅ 验收已通过，正在返回工作台...")
                                time.sleep(1.5)
                                App.rerun()
                            else:
                                App.error("提交失败，请重试")
                        except Exception as e:
                            App.error(f"保存验收数据失败: {str(e)}")
                else:
                    App.error("⚠️ AI 评分低于阈值，建议核实修复质量。")
                    if App.button("🚀 申请强制核准", type="primary", use_container_width=True):
                        App.session_state["show_force_dialog"] = True
                        App.rerun()
                
                # 无论评分高低，都提供“暂存返回”选项
                if App.button("⏳ 暂存待修 (返回工作台)", type="secondary", use_container_width=True, key="hold_task_btn"):
                    del App.session_state["active_wo_for_detect"]
                    if "last_analysis_res" in App.session_state: del App.session_state["last_analysis_res"]
                    App.session_state["active_tab"] = "🛠️ 我的工作台"
                    App.success("已取消验收，任务保留在处理中。")
                    time.sleep(1)
                    App.rerun()
                
                if App.session_state.get("show_force_dialog"):
                    with App.form("force_form"):
                        App.markdown("### 📝 申请强制核准")
                        reason = App.text_area("请说明申请强制核准的原因 (必填)", 
                                            placeholder="例如：AI 识别存在误差，现场管段已通过物理手段彻底清淤，环境复杂导致拍摄不佳等...",
                                            help="请详细描述现场实际情况，以便管理员审核通过")
                        
                        f_c1, f_c2 = App.columns(2)
                        submitted = f_c1.form_submit_button("🚀 提交申请", use_container_width=True)
                        cancelled = f_c2.form_submit_button("❌ 取消", use_container_width=True)
                        
                        if submitted:
                            if not reason or len(reason.strip()) < 5:
                                App.error("⚠️ 请填写详细的申请原因（至少5个字符）")
                            else:
                                active_wo = App.session_state.get("active_wo_for_detect")
                                user_info = App.session_state.get("user_info", {})
                                last_res = App.session_state.get("last_analysis_res")
                                
                                if active_wo and user_info and last_res:
                                    # 1. 保存当前的验收照片
                                    try:
                                        images_data = {"segmented": last_res["plot"]}
                                        task_id = active_wo.get("task_id") or 0
                                        result_id = detection_service.save_detection_result(
                                            task_id=task_id,
                                            frame_number=1,
                                            analysis_data=last_res.get("data", {}),
                                            images_data=images_data
                                        )
                                        
                                        # 2. 更新工单状态并关联 result_id
                                        success = workorder_service.update_work_order(
                                            active_wo["id"], 
                                            status="待审核", 
                                            is_forced=True,
                                            result_id=result_id,
                                            processing_notes=f"【强制核准申请】原因：{reason}"
                                        )
                                        
                                        if success:
                                            audit_service.log_action(
                                                user_info.get("id"), 
                                                user_info.get("username"), 
                                                "申请强制核准", "现场验收", f"任务 {wo_code} 申请强制核准。原因：{reason[:20]}..."
                                            )
                                            App.session_state["show_force_dialog"] = False
                                            del App.session_state["active_wo_for_detect"] 
                                            if "last_analysis_res" in App.session_state: del App.session_state["last_analysis_res"]
                                            App.session_state["active_tab"] = "🛠️ 我的工作台"
                                            App.success("✅ 申请已提交！任务状态已更新为“待验收”，请等待管理员审核。")
                                            time.sleep(1.5)
                                            App.rerun()
                                        else:
                                            App.error("提交失败，请检查网络或联系管理员")
                                    except Exception as e:
                                        App.error(f"保存数据失败: {str(e)}")
                                else:
                                    App.error("未找到分析结果或任务信息，请重新分析")
                        if cancelled:
                            App.session_state["show_force_dialog"] = False
                            App.rerun()

def show_worker_self_service():
    """现场工程师个人中心"""
    user_info = App.session_state.get("user_info", {})
    user_id = user_info.get("id")
    
    if not user_id:
        App.error("未找到用户信息")
        return

    user = user_service.get_user_by_id(user_id)
    if not user:
        App.error("无法获取个人资料")
        return

    profile_col, stats_col = App.columns([2, 3])

    with profile_col:
        with App.container(border=True):
            App.markdown("#### 🆔 基本信息")
            App.write(f"**用户名：** {user.username}")
            App.write(f"**姓名：** {user.full_name or '未填写'}")
            App.write(f"**联系电话：** {user.phone or '未填写'}")
            App.write(f"**岗位角色：** 现场工程师")
            
            if App.button("📝 修改资料", use_container_width=True):
                App.session_state["show_profile_edit"] = True
    
    with stats_col:
        with App.container(border=True):
            App.markdown("#### 📈 工作统计")
            stats = workorder_service.get_work_order_stats(assignee_id=user_id)
            status_dist = stats.get("status_distribution", {})
            
            c1, c2 = App.columns(2)
            c1.metric("累计完结任务", stats.get("closed", 0))
            # 这里的状态对应数据库中的 '进行中' (UI显示为处理中)
            pending_count = status_dist.get("进行中", 0) + status_dist.get("待指派", 0)
            c2.metric("当前待处理任务", pending_count)
            
            App.markdown("---")
            App.markdown("#### 🕒 最近操作记录")
            logs = audit_service.list_logs(user_id=user_id, limit=5)
            if logs:
                for log in logs:
                    time_str = log.created_at.strftime('%Y-%m-%d %H:%M') if log.created_at else "未知"
                    App.caption(f"{time_str}　{log.action}　（{log.module}）")
            else:
                App.info("暂无操作记录")

    # 编辑表单
    if App.session_state.get("show_profile_edit"):
        with App.form("profile_edit_form"):
            App.markdown("#### ✏️ 修改个人资料")
            new_full_name = App.text_input("姓名", value=user.full_name or "")
            new_phone = App.text_input("电话", value=user.phone or "")
            new_email = App.text_input("邮箱", value=user.email or "")
            new_pwd = App.text_input("修改密码 (不改请留空)", type="password")
            
            sc1, sc2 = App.columns(2)
            if sc1.form_submit_button("💾 保存更改", use_container_width=True):
                update_data = {
                    "full_name": new_full_name,
                    "phone": new_phone,
                    "email": new_email
                }
                if new_pwd:
                    update_data["password_hash"] = new_pwd
                
                if user_service.update_user(user_id, **update_data):
                    App.success("✅ 资料更新成功！")
                    audit_service.log_action(user_id, user.username, "更新资料", "个人中心", "用户修改了个人资料")
                    App.session_state["show_profile_edit"] = False
                    time.sleep(1)
                    App.rerun()
            if sc2.form_submit_button("❌ 取消", use_container_width=True):
                App.session_state["show_profile_edit"] = False
                App.rerun()

if __name__ == "__main__":
    run_app()
