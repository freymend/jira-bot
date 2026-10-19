from __future__ import annotations

import logging
import re

# Attendee only accepts the screen-share websocket credential as a URL query parameter,
# and uvicorn logs request paths with their query strings.
_TOKEN_QUERY = re.compile(r"([?&]token=)[^&\s\"']+", re.IGNORECASE)
_REDACTED_LOGGERS = ("uvicorn.access", "uvicorn.error")


def redact_query_tokens(value: str) -> str:
    return _TOKEN_QUERY.sub(r"\1<redacted>", value)


class QueryTokenRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_query_tokens(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                redact_query_tokens(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def install_query_token_redaction() -> None:
    for name in _REDACTED_LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(f, QueryTokenRedactionFilter) for f in logger.filters):
            logger.addFilter(QueryTokenRedactionFilter())
