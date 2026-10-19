import logging
import unittest

from app.log_redaction import (
    QueryTokenRedactionFilter,
    install_query_token_redaction,
    redact_query_tokens,
)


class QueryTokenRedactionTests(unittest.TestCase):
    def test_redacts_token_query_values_only(self) -> None:
        self.assertEqual(
            redact_query_tokens("/webhooks/attendee/video?token=secret-value&x=1"),
            "/webhooks/attendee/video?token=<redacted>&x=1",
        )
        self.assertEqual(
            redact_query_tokens("/meetings?bot=1&TOKEN=secret"),
            "/meetings?bot=1&TOKEN=<redacted>",
        )
        self.assertEqual(redact_query_tokens("/health"), "/health")

    def test_uvicorn_websocket_log_line_is_redacted(self) -> None:
        record = logging.LogRecord(
            "uvicorn.error",
            logging.INFO,
            __file__,
            1,
            '%s - "WebSocket %s" [accepted]',
            ("172.18.0.10:57180", "/webhooks/attendee/video?token=secret-value"),
            None,
        )
        self.assertTrue(QueryTokenRedactionFilter().filter(record))
        message = record.getMessage()
        self.assertNotIn("secret-value", message)
        self.assertIn("/webhooks/attendee/video?token=<redacted>", message)

    def test_install_adds_one_filter_per_uvicorn_logger(self) -> None:
        install_query_token_redaction()
        install_query_token_redaction()
        for name in ("uvicorn.access", "uvicorn.error"):
            filters = logging.getLogger(name).filters
            self.assertEqual(
                sum(isinstance(f, QueryTokenRedactionFilter) for f in filters), 1
            )
