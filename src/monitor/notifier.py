"""
通知发送 — Telegram / DingTalk

匹配 engine.py 中的调用：
- notifier.send(message: str) -> None
"""

from __future__ import annotations

from typing import Optional

import aiohttp
from loguru import logger


class Notifier:
    """
    通知发送器

    支持：
    1. Telegram Bot
    2. DingTalk Webhook
    3. 控制台（fallback）
    """

    def __init__(self, config: dict):
        self._config = config

        # Telegram 配置
        tg_cfg = config.get("telegram", {})
        self._tg_enabled = tg_cfg.get("enabled", False)
        self._tg_token = tg_cfg.get("bot_token", "")
        self._tg_chat_id = tg_cfg.get("chat_id", "")

        # DingTalk 配置
        dd_cfg = config.get("dingtalk", {})
        self._dd_enabled = dd_cfg.get("enabled", False)
        self._dd_webhook = dd_cfg.get("webhook_url", "")

    async def send(self, message: str) -> None:
        """
        发送通知消息

        同时发送到所有启用的渠道
        """
        logger.info(f"Notification: {message}")

        if self._tg_enabled:
            await self._send_telegram(message)

        if self._dd_enabled:
            await self._send_dingtalk(message)

    async def _send_telegram(self, message: str) -> None:
        """发送 Telegram 消息"""
        if not self._tg_token or not self._tg_chat_id:
            return

        url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
        payload = {
            "chat_id": self._tg_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Telegram send failed: {resp.status} {body}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def _send_dingtalk(self, message: str) -> None:
        """发送 DingTalk 消息"""
        if not self._dd_webhook:
            return

        payload = {
            "msgtype": "text",
            "text": {"content": f"[Trading] {message}"},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._dd_webhook, json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"DingTalk send failed: {resp.status}")
        except Exception as e:
            logger.error(f"DingTalk send error: {e}")
