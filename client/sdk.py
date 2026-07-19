"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterator
from typing import Any

import requests


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        self.status_code = status_code
        self.message = message
        self.request_id = request_id

        details = message

        if status_code is not None:
            details = f"HTTP {status_code}: {details}"

        if request_id:
            details += f" [request_id={request_id}]"

        super().__init__(details)


class DocumentAnalystClient:
    """Client for a Databricks Document Analyst serving endpoint."""

    RETRYABLE_STATUS_CODES = {429, 503}

    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not endpoint_name.strip():
            raise ValueError("endpoint_name cannot be empty.")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        resolved_host = host or os.getenv("DATABRICKS_HOST")
        resolved_token = token or os.getenv("DATABRICKS_TOKEN")

        if not resolved_host:
            raise ValueError(
                "Pass host or set DATABRICKS_HOST."
            )

        if not resolved_token:
            raise ValueError(
                "Pass token or set DATABRICKS_TOKEN."
            )

        if not resolved_host.startswith(
            ("http://", "https://")
        ):
            resolved_host = f"https://{resolved_host}"

        self.endpoint_name = endpoint_name
        self.host = resolved_host.rstrip("/")
        self.token = resolved_token
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

        self.invocation_url = (
            f"{self.host}/serving-endpoints/"
            f"{self.endpoint_name}/invocations"
        )

        self.status_url = (
            f"{self.host}/api/2.0/serving-endpoints/"
            f"{self.endpoint_name}"
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    @staticmethod
    def _get_request_id(
        response: requests.Response,
    ) -> str | None:
        return (
            response.headers.get("x-request-id")
            or response.headers.get(
                "x-databricks-request-id"
            )
            or response.headers.get("trace-id")
        )

    @staticmethod
    def _get_error_message(
        response: requests.Response,
    ) -> str:
        try:
            data = response.json()
        except ValueError:
            return (
                response.text.strip()
                or response.reason
                or "Unknown endpoint error."
            )

        if isinstance(data, dict):
            for key in (
                "message",
                "error",
                "detail",
                "error_code",
            ):
                value = data.get(key)

                if value:
                    if isinstance(value, (dict, list)):
                        return json.dumps(value)

                    return str(value)

        return json.dumps(data)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)

                elif isinstance(block, dict):
                    text = (
                        block.get("text")
                        or block.get("content")
                    )

                    if text:
                        parts.append(str(text))

            return "".join(parts)

        if content is None:
            return ""

        return str(content)

    @classmethod
    def _extract_answer(cls, data: Any) -> str:
        """Extract the final answer from supported response shapes."""

        # Current Path A endpoint:
        # [
        #   {
        #       "messages": [...]
        #   }
        # ]
        if isinstance(data, list) and data:
            first = data[0]

            if isinstance(first, dict):
                messages = first.get("messages", [])

                if messages:
                    last = messages[-1]

                    if isinstance(last, dict):
                        content = cls._content_to_text(
                            last.get("content")
                        ).strip()

                        if content:
                            return content

                    content = cls._content_to_text(
                        getattr(last, "content", last)
                    ).strip()

                    if content:
                        return content

                final_answer = first.get("final_answer")

                if final_answer:
                    return str(final_answer).strip()

        if isinstance(data, dict):
            # MLflow prediction wrapper.
            if "predictions" in data:
                return cls._extract_answer(
                    data["predictions"]
                )

            # OpenAI-compatible response.
            choices = data.get("choices", [])

            if choices:
                first_choice = choices[0]

                if isinstance(first_choice, dict):
                    message = first_choice.get(
                        "message",
                        {},
                    )

                    if isinstance(message, dict):
                        content = cls._content_to_text(
                            message.get("content")
                        ).strip()

                        if content:
                            return content

            # Standard MLflow ChatAgent response.
            messages = data.get("messages", [])

            if messages:
                # The response may include tool and assistant
                # messages. Search backwards for final assistant text.
                for message in reversed(messages):
                    if not isinstance(message, dict):
                        continue

                    role = message.get("role")
                    content = cls._content_to_text(
                        message.get("content")
                    ).strip()

                    if role == "assistant" and content:
                        return content

                last = messages[-1]

                if isinstance(last, dict):
                    content = cls._content_to_text(
                        last.get("content")
                    ).strip()

                    if content:
                        return content

            final_answer = data.get("final_answer")

            if final_answer:
                return str(final_answer).strip()

            output = data.get("output")

            if isinstance(output, str) and output.strip():
                return output.strip()

        raise AnalystClientError(
            status_code=None,
            message=(
                "Could not find a final answer in the "
                "endpoint response."
            ),
        )

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        return (2**attempt) + random.uniform(0, 0.25)

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> requests.Response:
        started_at = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )

            except requests.Timeout as exc:
                elapsed = (
                    time.perf_counter() - started_at
                )

                raise TimeoutError(
                    "Document Analyst request timed out "
                    f"after {elapsed:.2f} seconds. "
                    f"Configured timeout: {self.timeout}s."
                ) from exc

            except requests.RequestException as exc:
                raise AnalystClientError(
                    status_code=None,
                    message=(
                        "Could not connect to the endpoint: "
                        f"{exc}"
                    ),
                ) from exc

            if response.status_code < 400:
                return response

            retryable = (
                response.status_code
                in self.RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            )

            if retryable:
                delay = self._backoff_seconds(attempt)

                print(
                    f"Attempt {attempt + 1} returned "
                    f"HTTP {response.status_code}. "
                    f"Retrying in {delay:.2f}s..."
                )

                response.close()
                time.sleep(delay)
                continue

            raise AnalystClientError(
                status_code=response.status_code,
                message=self._get_error_message(response),
                request_id=self._get_request_id(response),
            )

        raise AnalystClientError(
            status_code=None,
            message="All request attempts failed.",
        )

    def health_check(self) -> bool:
        """Return True when the endpoint is READY."""
        try:
            response = self._request(
                "GET",
                self.status_url,
            )
        except AnalystClientError:
            return False

        try:
            data = response.json()
        except ValueError:
            return False

        state = data.get("state", {})

        if not isinstance(state, dict):
            return False

        ready = str(
            state.get("ready", "")
        ).upper()

        config_update = str(
            state.get("config_update", "")
        ).upper()

        return (
            ready == "READY"
            and config_update
            not in {
                "IN_PROGRESS",
                "UPDATE_FAILED",
            }
        )

    def ask(self, question: str) -> str:
        """Send a question and return its final answer."""
        if not question or not question.strip():
            raise ValueError("question cannot be empty.")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": question.strip(),
                }
            ]
        }

        response = self._request(
            "POST",
            self.invocation_url,
            payload=payload,
        )

        try:
            data = response.json()
        except ValueError as exc:
            raise AnalystClientError(
                status_code=response.status_code,
                message=(
                    "Endpoint returned invalid JSON: "
                    f"{response.text[:500]}"
                ),
                request_id=self._get_request_id(response),
            ) from exc

        return self._extract_answer(data)

    @classmethod
    def _extract_stream_text(
        cls,
        event: Any,
    ) -> str | None:
        if isinstance(event, str):
            return event

        if not isinstance(event, dict):
            return None

        # Standard ChatAgent chunk:
        # {"delta": {"role": "assistant", "content": "..."}}
        delta = event.get("delta")

        if isinstance(delta, dict):
            content = cls._content_to_text(
                delta.get("content")
            )

            if content:
                return content

        if isinstance(delta, str) and delta:
            return delta

        # OpenAI streaming shape.
        choices = event.get("choices", [])

        if choices:
            choice = choices[0]

            if isinstance(choice, dict):
                choice_delta = choice.get("delta", {})

                if isinstance(choice_delta, dict):
                    content = cls._content_to_text(
                        choice_delta.get("content")
                    )

                    if content:
                        return content

        for key in ("content", "text", "token"):
            value = event.get(key)

            if isinstance(value, str) and value:
                return value

        return None

    def ask_streaming(self,question: str,) -> Iterator[str]:
        """Yield streaming chunks, or the full answer once if unsupported."""

        if not question or not question.strip():
            raise ValueError("question cannot be empty.")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": question.strip(),
                }
            ],
            "stream": True,
        }

        try:
            response = self._request(
                "POST",
                self.invocation_url,
                payload=payload,
                stream=True,
            )

        except AnalystClientError as exc:
            error_message = str(exc).lower()

            # Bare LangGraph models logged with mlflow.langchain.log_model
            # may reject stream=True because predict_stream is not implemented.
            if (
                exc.status_code == 400
                and (
                    "does not support streaming" in error_message
                    or "streaming" in error_message
                )
            ):
                yield self.ask(question)
                return

            raise

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        yielded = False

        if (
            "text/event-stream" in content_type
            or "application/x-ndjson" in content_type
        ):
            for raw_line in response.iter_lines(
                decode_unicode=True
            ):
                if not raw_line:
                    continue

                line = raw_line.strip()

                if line.startswith("data:"):
                    line = line[5:].strip()

                if not line or line == "[DONE]":
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    yielded = True
                    continue

                text = self._extract_stream_text(event)

                if text:
                    yield text
                    yielded = True

            if yielded:
                return

        # The endpoint accepted the request but returned ordinary JSON.
        try:
            data = response.json()
        except ValueError as exc:
            raise AnalystClientError(
                status_code=response.status_code,
                message=(
                    "Streaming response was neither SSE "
                    "nor valid JSON."
                ),
                request_id=self._get_request_id(response),
            ) from exc

        answer = self._extract_answer(data)

        if answer:
            yield answer