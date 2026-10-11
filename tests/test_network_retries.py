from __future__ import annotations

import unittest

import httpx

from app.network import (
    is_ambiguous_write_error,
    is_safe_to_retry_write,
    is_transient_network_error,
    retry_after_seconds,
)


class NetworkRetryTests(unittest.TestCase):
    def test_transport_failures_are_retryable(self) -> None:
        request = httpx.Request("GET", "https://example.test/transcript")
        self.assertTrue(
            is_transient_network_error(httpx.ConnectError("offline", request=request))
        )
        self.assertTrue(
            is_transient_network_error(httpx.ReadTimeout("slow", request=request))
        )

    def test_retryable_status_and_retry_after_are_detected(self) -> None:
        request = httpx.Request("GET", "https://example.test/transcript")
        response = httpx.Response(
            429,
            headers={"Retry-After": "17"},
            request=request,
        )
        error = httpx.HTTPStatusError(
            "rate limited", request=request, response=response
        )

        self.assertTrue(is_transient_network_error(error))
        self.assertEqual(retry_after_seconds(error), 17)

    def test_permanent_client_errors_are_not_retried(self) -> None:
        request = httpx.Request("GET", "https://example.test/transcript")
        response = httpx.Response(401, request=request)
        error = httpx.HTTPStatusError(
            "unauthorized", request=request, response=response
        )

        self.assertFalse(is_transient_network_error(error))
        self.assertIsNone(retry_after_seconds(error))
        self.assertFalse(is_ambiguous_write_error(error))

    def test_only_definitely_unsent_or_rate_limited_writes_are_retried(self) -> None:
        request = httpx.Request("POST", "https://jira.example/issue")
        connect_error = httpx.ConnectError("offline", request=request)
        limited_response = httpx.Response(429, request=request)
        limited_error = httpx.HTTPStatusError(
            "rate limited", request=request, response=limited_response
        )
        timeout_error = httpx.ReadTimeout("response lost", request=request)

        self.assertTrue(is_safe_to_retry_write(connect_error))
        self.assertTrue(is_safe_to_retry_write(limited_error))
        self.assertFalse(is_safe_to_retry_write(timeout_error))
        self.assertTrue(is_ambiguous_write_error(timeout_error))


if __name__ == "__main__":
    unittest.main()
