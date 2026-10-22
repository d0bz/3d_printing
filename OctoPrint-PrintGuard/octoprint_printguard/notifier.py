# coding=utf-8
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional
import requests

from .models import DetectionResult, PrinterContext

logger = logging.getLogger("octoprint.plugins.printguard.notifier")


class NotificationManager:
    """Dispatches print failure alerts and snapshots to Discord, Telegram, Pushover, and Generic Webhooks."""

    def __init__(self, settings_getter: Any = None, timeout: int = 12):
        self._settings_getter = settings_getter
        self.timeout = timeout

    def _get(self, key: str, default: Any = None) -> Any:
        if callable(self._settings_getter):
            val = self._settings_getter(key)
            return default if val is None else val
        if isinstance(self._settings_getter, dict):
            return self._settings_getter.get(key, default)
        return default

    def notify(
        self,
        result: DetectionResult,
        snapshot_jpeg: Optional[bytes] = None,
        context: Optional[PrinterContext] = None,
    ) -> Dict[str, bool]:
        """Send notifications across all enabled channels and return per-channel status."""
        outcomes: Dict[str, bool] = {}

        if self._get("discord_enabled", False) and self._get("discord_webhook_url", ""):
            outcomes["discord"] = self.send_discord(
                webhook_url=str(self._get("discord_webhook_url")),
                result=result,
                snapshot_jpeg=snapshot_jpeg,
                context=context,
            )

        if (
            self._get("telegram_enabled", False)
            and self._get("telegram_bot_token", "")
        ):
            outcomes["telegram"] = self.send_telegram(
                bot_token=str(self._get("telegram_bot_token")).strip(),
                chat_id=str(self._get("telegram_chat_id", "")).strip(),
                result=result,
                snapshot_jpeg=snapshot_jpeg,
                context=context,
            )

        if (
            self._get("pushover_enabled", False)
            and self._get("pushover_api_token", "")
            and self._get("pushover_user_key", "")
        ):
            outcomes["pushover"] = self.send_pushover(
                api_token=str(self._get("pushover_api_token")),
                user_key=str(self._get("pushover_user_key")),
                result=result,
                snapshot_jpeg=snapshot_jpeg,
                context=context,
            )

        if self._get("webhook_enabled", False) and self._get("webhook_url", ""):
            outcomes["webhook"] = self.send_generic_webhook(
                webhook_url=str(self._get("webhook_url")),
                webhook_secret=str(self._get("webhook_secret", "")),
                result=result,
                snapshot_jpeg=snapshot_jpeg,
                context=context,
            )

        return outcomes

    @staticmethod
    def _format_summary_text(
        result: DetectionResult, context: Optional[PrinterContext] = None
    ) -> str:
        title = (
            "🚨 3D Print Failure Detected!"
            if result.is_failure
            else "⚠️ 3D Print Warning Detected"
        )
        failures = ", ".join(result.failure_labels) if result.failure_labels else "Anomaly"
        lines = [
            title,
            f"• Type: {failures}",
            f"• Confidence: {result.confidence * 100:.0f}% ({result.source})",
            f"• Details: {result.explanation}",
        ]
        if result.action_taken:
            lines.append(f"• Action Taken: {result.action_taken}")
        if context:
            if context.filename:
                lines.append(f"• File: {context.filename}")
            if context.current_z is not None:
                lines.append(f"• Z Height: {context.current_z:.2f} mm")
            if context.progress_completion:
                lines.append(f"• Progress: {context.progress_completion:.1f}%")
        return "\n".join(lines)

    def send_discord(
        self,
        webhook_url: str,
        result: DetectionResult,
        snapshot_jpeg: Optional[bytes] = None,
        context: Optional[PrinterContext] = None,
    ) -> bool:
        try:
            color = 0xD22323 if result.is_failure else 0xE1910F
            fields: List[Dict[str, Any]] = [
                {
                    "name": "Failure Modes",
                    "value": ", ".join(result.failure_labels) or "None",
                    "inline": True,
                },
                {
                    "name": "Confidence",
                    "value": f"{result.confidence * 100:.0f}% ({result.source})",
                    "inline": True,
                },
            ]
            if result.action_taken:
                fields.append(
                    {"name": "Action Taken", "value": result.action_taken, "inline": True}
                )
            if context and context.current_z is not None:
                fields.append(
                    {"name": "Z Height", "value": f"{context.current_z:.2f} mm", "inline": True}
                )

            embed: Dict[str, Any] = {
                "title": "PrintGuard Alert: Print Failure Detected"
                if result.is_failure
                else "PrintGuard Warning",
                "description": result.explanation,
                "color": color,
                "fields": fields,
            }

            if snapshot_jpeg:
                embed["image"] = {"url": "attachment://snapshot.jpg"}
                payload_json = json.dumps({"embeds": [embed]})
                files = {
                    "file": ("snapshot.jpg", snapshot_jpeg, "image/jpeg"),
                }
                resp = requests.post(
                    webhook_url,
                    data={"payload_json": payload_json},
                    files=files,
                    timeout=self.timeout,
                )
            else:
                resp = requests.post(
                    webhook_url,
                    json={"embeds": [embed]},
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Discord webhook notification failed: %s", exc)
            return False

    def resolve_telegram_chat_id(self, bot_token: str) -> str:
        """Auto-discover chat_id from Telegram getUpdates if not explicitly configured."""
        base_api = f"https://api.telegram.org/bot{bot_token}"
        resp = requests.get(f"{base_api}/getUpdates", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        for item in reversed(data.get("result") or []):
            for key in ("message", "edited_message", "channel_post", "my_chat_member"):
                msg = item.get(key) or {}
                chat = msg.get("chat") or {}
                if chat.get("id") is not None:
                    return str(chat["id"])
        return ""

    def send_telegram(
        self,
        bot_token: str,
        chat_id: str,
        result: DetectionResult,
        snapshot_jpeg: Optional[bytes] = None,
        context: Optional[PrinterContext] = None,
    ) -> bool:
        try:
            resolved_chat_id = (chat_id or "").strip()
            if not resolved_chat_id:
                resolved_chat_id = self.resolve_telegram_chat_id(bot_token)
            if not resolved_chat_id:
                logger.error(
                    "Telegram chat_id is empty and no recent messages found in getUpdates. "
                    "Send /start to your Telegram bot first or enter Chat ID in settings."
                )
                return False

            caption = self._format_summary_text(result, context)
            base_api = f"https://api.telegram.org/bot{bot_token}"
            if snapshot_jpeg:
                resp = requests.post(
                    f"{base_api}/sendPhoto",
                    data={"chat_id": resolved_chat_id, "caption": caption},
                    files={"photo": ("snapshot.jpg", snapshot_jpeg, "image/jpeg")},
                    timeout=self.timeout,
                )
            else:
                resp = requests.post(
                    f"{base_api}/sendMessage",
                    json={"chat_id": resolved_chat_id, "text": caption},
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Telegram notification failed: %s", exc)
            return False

    def send_pushover(
        self,
        api_token: str,
        user_key: str,
        result: DetectionResult,
        snapshot_jpeg: Optional[bytes] = None,
        context: Optional[PrinterContext] = None,
    ) -> bool:
        try:
            message = self._format_summary_text(result, context)
            data = {
                "token": api_token,
                "user": user_key,
                "title": "OctoPrint PrintGuard Alert",
                "message": message,
                "priority": "1" if result.is_failure else "0",
            }
            files = (
                {"attachment": ("snapshot.jpg", snapshot_jpeg, "image/jpeg")}
                if snapshot_jpeg
                else None
            )
            resp = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Pushover notification failed: %s", exc)
            return False

    def send_generic_webhook(
        self,
        webhook_url: str,
        webhook_secret: str,
        result: DetectionResult,
        snapshot_jpeg: Optional[bytes] = None,
        context: Optional[PrinterContext] = None,
    ) -> bool:
        try:
            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["X-PrintGuard-Secret"] = webhook_secret

            payload: Dict[str, Any] = {
                "event": "printguard_alert",
                "detection": result.to_dict(),
                "printer": context.to_dict() if context else {},
            }
            if snapshot_jpeg:
                payload["snapshot_base64"] = base64.b64encode(snapshot_jpeg).decode("ascii")

            resp = requests.post(
                webhook_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Generic webhook notification failed: %s", exc)
            return False
