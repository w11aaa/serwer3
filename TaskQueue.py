import threading
import queue
import time
import os
from datetime import datetime
from database import db_ops
from db_service import detection_service

class BackgroundTaskWorker:
    def __init__(self, max_workers=2):
        self.task_queue = queue.Queue()
        self.workers = []
        self.max_workers = max_workers
        self.running = True
        self._start_workers()

    def _start_workers(self):
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

    def _worker_loop(self):
        while self.running:
            try:
                # 阻塞获取任务，超时 1 秒以便检查 self.running
                task_func, task_id, args, kwargs = self.task_queue.get(timeout=1.0)
                
                try:
                    # 更新任务状态为处理中
                    detection_service.update_task_status(task_id, "processing", 0.0)
                    
                    # 执行任务
                    task_func(task_id, *args, **kwargs)
                    
                    # 状态由任务函数内部更新为已完成或失败
                except Exception as e:
                    print(f"Task {task_id} failed: {str(e)}")
                    detection_service.update_task_status(task_id, "failed", 0.0)
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def add_task(self, task_func, task_id, *args, **kwargs):
        self.task_queue.put((task_func, task_id, args, kwargs))

    def stop(self):
        self.running = False
        for t in self.workers:
            t.join(timeout=1.0)

# 全局任务队列实例
task_worker = BackgroundTaskWorker(max_workers=1) # 视频处理较重，默认 1 个并行即可
