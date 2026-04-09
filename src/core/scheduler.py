"""
定时任务调度器 — 处理周期性任务（重平衡、健康检查、数据清理）
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from loguru import logger


@dataclass
class ScheduledTask:
    """定时任务"""
    name: str
    interval_seconds: float
    callback: Callable[[], Awaitable[None]]
    _running: bool = field(default=False, init=False)


class Scheduler:
    """异步定时任务调度器"""

    def __init__(self):
        self._tasks: list[ScheduledTask] = []
        self._running = False
        self._async_tasks: list[asyncio.Task] = []

    def add_task(
        self,
        name: str,
        interval_seconds: float,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """注册定时任务"""
        self._tasks.append(
            ScheduledTask(
                name=name,
                interval_seconds=interval_seconds,
                callback=callback,
            )
        )
        logger.info(
            f"Scheduled task registered: {name} "
            f"(every {interval_seconds}s)"
        )

    async def start(self) -> None:
        """启动所有定时任务"""
        self._running = True
        for task in self._tasks:
            t = asyncio.create_task(self._run_task(task))
            self._async_tasks.append(t)
        logger.info(f"Scheduler started with {len(self._tasks)} tasks")

    async def stop(self) -> None:
        """停止所有定时任务"""
        self._running = False
        for t in self._async_tasks:
            t.cancel()
        await asyncio.gather(*self._async_tasks, return_exceptions=True)
        self._async_tasks.clear()
        logger.info("Scheduler stopped")

    async def _run_task(self, task: ScheduledTask) -> None:
        """执行单个定时任务的循环"""
        while self._running:
            try:
                await asyncio.sleep(task.interval_seconds)
                if not self._running:
                    break
                logger.debug(f"Running scheduled task: {task.name}")
                await task.callback()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled task error [{task.name}]: {e}")
