"""Email delivery of the daily report via smtplib."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from daytrader.config.settings import Secrets
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_message(
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> EmailMessage:
    """Construct a MIME email with optional file attachments (pure/testable)."""
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    for path in attachments or []:
        path = Path(path)
        if not path.exists():
            logger.warning("Attachment missing, skipping: %s", path)
            continue
        data = path.read_bytes()
        subtype = "pdf" if path.suffix.lower() == ".pdf" else "octet-stream"
        maintype = "application" if subtype != "png" else "image"
        if path.suffix.lower() == ".png":
            maintype, subtype = "image", "png"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)
    return msg


class EmailSender:
    """Sends reports over SMTP (STARTTLS). No-ops with a warning if unconfigured."""

    def __init__(self, secrets: Secrets) -> None:
        self.secrets = secrets

    @property
    def is_configured(self) -> bool:
        s = self.secrets
        return bool(s.smtp_host and s.smtp_username and s.smtp_password and s.email_from and s.email_to)

    def send(self, subject: str, body: str, attachments: list[Path] | None = None) -> bool:
        if not self.is_configured:
            logger.warning("Email not configured (missing SMTP settings); skipping send.")
            return False
        s = self.secrets
        msg = build_message(s.email_from, s.email_to, subject, body, attachments)
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as server:
                server.starttls(context=context)
                server.login(s.smtp_username, s.smtp_password)
                server.send_message(msg)
            logger.info("Report emailed to %s", s.email_to)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send report email")
            return False
