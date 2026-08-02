import json
import logging
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

JSON_MIME_TYPE = "application/json"
_runtime_lock = threading.Lock()
_runtimes: dict[str, "TracingRuntime"] = {}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _set_json_input(span: Span, value: Any) -> None:
    span.set_attribute("input.value", _json(value))
    span.set_attribute("input.mime_type", JSON_MIME_TYPE)


def _set_json_output(span: Span, value: Any) -> None:
    span.set_attribute("output.value", _json(value))
    span.set_attribute("output.mime_type", JSON_MIME_TYPE)


def _set_attributes(span: Span, attributes: Mapping[str, Any] | None) -> None:
    for key, value in (attributes or {}).items():
        if value is not None:
            span.set_attribute(key, value)


@dataclass
class TracingRuntime:
    provider: Any
    tracer: Any
    project_name: str

    def force_flush(self) -> None:
        try:
            force_flush = getattr(self.provider, "force_flush", None)
            if force_flush is not None:
                force_flush()
        except Exception as exc:  # observability must not fail data generation
            logger.warning("Phoenix trace flush failed: %s", exc)


@dataclass
class ObjectTrace:
    span: Span
    carrier: dict[str, str]
    trace_id: str

    def set_output(self, value: Any) -> None:
        _set_json_output(self.span, value)

    def mark_error(self, error: BaseException | str) -> None:
        message = str(error)
        if isinstance(error, BaseException):
            self.span.record_exception(error)
        self.span.set_attribute("error.message", message)
        self.span.set_status(Status(StatusCode.ERROR, message))


def configure_tracing(
    *,
    provider: Any | None = None,
    project_name: str | None = None,
) -> TracingRuntime:
    """Configure Phoenix once, or install an explicit provider for tests."""
    with _runtime_lock:
        resolved_project = project_name or os.getenv(
            "PHOENIX_PROJECT_NAME",
            "search-query-generation",
        )
        if provider is not None:
            runtime = TracingRuntime(
                provider=provider,
                tracer=provider.get_tracer("data_pipelines"),
                project_name=resolved_project,
            )
            _runtimes[resolved_project] = runtime
            return runtime
        if resolved_project in _runtimes:
            return _runtimes[resolved_project]
        endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://127.0.0.1:6006",
        ).rstrip("/")
        try:
            from phoenix.otel import register

            resolved_provider = register(
                project_name=resolved_project,
                endpoint=f"{endpoint}/v1/traces",
                auto_instrument=True,
                batch=True,
                verbose=False,
            )
            tracer = resolved_provider.get_tracer("data_pipelines")
        except Exception as exc:
            logger.warning(
                "Phoenix tracing is unavailable; continuing without export: %s",
                exc,
            )
            resolved_provider = trace.get_tracer_provider()
            tracer = trace.get_tracer("data_pipelines")

        runtime = TracingRuntime(
            provider=resolved_provider,
            tracer=tracer,
            project_name=resolved_project,
        )
        _runtimes[resolved_project] = runtime
        return runtime


def get_tracing(project_name: str | None = None) -> TracingRuntime:
    return configure_tracing(project_name=project_name)


@contextmanager
def object_trace(
    name: str,
    input_value: Any,
    attributes: Mapping[str, Any],
    project_name: str | None = None,
) -> Iterator[ObjectTrace]:
    runtime = get_tracing(project_name)
    span = runtime.tracer.start_span(
        name,
        attributes={
            "openinference.span.kind": "CHAIN",
            **{key: value for key, value in attributes.items() if value is not None},
        },
    )
    _set_json_input(span, input_value)
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(
        carrier,
        context=trace.set_span_in_context(span),
    )
    span_context = span.get_span_context()
    handle = ObjectTrace(
        span=span,
        carrier=carrier,
        trace_id=f"{span_context.trace_id:032x}",
    )
    try:
        yield handle
        if span.status.status_code == StatusCode.UNSET:
            span.set_status(Status(StatusCode.OK))
    except BaseException as exc:
        handle.mark_error(exc)
        raise
    finally:
        span.end()


@contextmanager
def stage_span(
    carrier: Mapping[str, str],
    name: str,
    kind: str,
    input_value: Any,
    attributes: Mapping[str, Any] | None = None,
    project_name: str | None = None,
) -> Iterator[Span]:
    runtime = get_tracing(project_name)
    parent_context = TraceContextTextMapPropagator().extract(carrier=carrier)
    span = runtime.tracer.start_span(
        name,
        context=parent_context,
        attributes={"openinference.span.kind": kind},
    )
    _set_attributes(span, attributes)
    _set_json_input(span, input_value)
    try:
        with trace.use_span(span, end_on_exit=False):
            yield span
        if span.status.status_code == StatusCode.UNSET:
            span.set_status(Status(StatusCode.OK))
    except BaseException as exc:
        span.record_exception(exc)
        span.set_attribute("error.message", str(exc))
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        span.end()


def set_span_output(span: Span, value: Any) -> None:
    _set_json_output(span, value)


def set_span_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
    _set_attributes(span, attributes)


def set_span_error(span: Span, message: str) -> None:
    span.set_attribute("error.message", message)
    span.set_status(Status(StatusCode.ERROR, message))
