from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from scripts.check_model_apis import (
    ModelApiConfig,
    chat_completion_url,
    extract_assistant_message,
    is_content_moderation_error,
    load_model_api_configs,
    post_json,
    should_retry_token_field,
    should_retry_transient_failure,
)


_USE_BOUND_RESPONSE_FORMAT = object()


def make_llm_client(
    config: ModelApiConfig,
    *,
    client_label: str | None = None,
    response_format: dict[str, Any] | None = None,
    max_output_tokens: int = 1200,
    content_filter_fallback_config: ModelApiConfig | None = None,
):
    """Build a metrics-aware client for an OpenAI-compatible chat endpoint."""

    metrics: dict[str, float | int] = {
        "llm_calls": 0,
        "successful_llm_calls": 0,
        "failed_llm_calls": 0,
        "http_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_token_reports": 0,
        "completion_token_reports": 0,
        "total_token_reports": 0,
        "token_usage_reported_calls": 0,
        "content_filter_fallback_calls": 0,
        "content_filter_fallback_successes": 0,
        "content_filter_fallback_failures": 0,
        "llm_time_seconds": 0.0,
    }
    fallback_client = (
        make_llm_client(
            content_filter_fallback_config,
            client_label=client_label,
            response_format=response_format,
            max_output_tokens=(
                content_filter_fallback_config.max_output_tokens_by_stage.get(
                    client_label or "unspecified",
                    max_output_tokens,
                )
            ),
        )
        if content_filter_fallback_config is not None
        else None
    )

    def merge_fallback_metrics(before: dict[str, Any]) -> None:
        if fallback_client is None:
            return
        delta = llm_metrics_delta(
            before,
            llm_metrics_snapshot(fallback_client),
        )
        for name in (
            "http_requests",
            "token_usage_reported_calls",
            "prompt_token_reports",
            "completion_token_reports",
            "total_token_reports",
        ):
            metrics[name] += int(delta.get(name, 0) or 0)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = delta.get(name)
            if value is not None:
                metrics[name] += int(value)

    def complete(
        prompt: str,
        response_format_override: dict[str, Any] | None | object = (
            _USE_BOUND_RESPONSE_FORMAT
        ),
    ) -> str:
        call_started = time.perf_counter()
        metrics["llm_calls"] += 1
        url = chat_completion_url(config)
        token_fields = ("max_tokens", "max_completion_tokens", None)
        last_error = "Unknown model API error."
        stage = client_label or "unspecified"

        for request_attempt in range(max(0, config.max_retries) + 1):
            retry_api_failure = False
            for token_field in token_fields:
                payload: dict[str, Any] = {
                    "model": config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0,
                }
                if config.enable_thinking is not None:
                    payload["enable_thinking"] = config.enable_thinking
                if token_field:
                    payload[token_field] = max(1, int(max_output_tokens))
                active_response_format = (
                    response_format
                    if response_format_override is _USE_BOUND_RESPONSE_FORMAT
                    else response_format_override
                )
                if active_response_format is not None:
                    if (
                        config.structured_output_mode == "json_object"
                        and active_response_format.get("type") == "json_schema"
                    ):
                        # Providers with JSON mode still need the original schema
                        # in the prompt; downstream semantic validators stay intact.
                        schema = active_response_format["json_schema"]["schema"]
                        payload["response_format"] = {"type": "json_object"}
                        payload["messages"][0]["content"] = (
                            prompt
                            + "\nReturn only a JSON object conforming to this schema:\n"
                            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
                        )
                    else:
                        payload["response_format"] = active_response_format

                metrics["http_requests"] += 1
                status_code, body, error = post_json(
                    url,
                    payload,
                    api_key=config.api_key if config.requires_api_key else None,
                    timeout=config.timeout,
                )
                if error is None:
                    content = extract_assistant_message(body)
                    if content:
                        usage = extract_token_usage(body)
                        if usage:
                            metrics["token_usage_reported_calls"] += 1
                            for name in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                            ):
                                value = usage.get(name)
                                if value is not None:
                                    metrics[name] += value
                                    report = f"{name.removesuffix('s')}_reports"
                                    metrics[report] += 1
                        metrics["successful_llm_calls"] += 1
                        metrics["llm_time_seconds"] += (
                            time.perf_counter() - call_started
                        )
                        return content
                    last_error = (
                        f"stage={stage} status={status_code}: "
                        "Model returned an empty assistant message."
                    )
                    retry_api_failure = True
                    break

                last_error = f"stage={stage} status={status_code}: {error}"
                if (
                    status_code == 400
                    and token_field is not None
                    and should_retry_token_field(error)
                ):
                    continue
                retry_api_failure = should_retry_transient_failure(
                    status_code,
                    error,
                )
                break

            if retry_api_failure and request_attempt < max(0, config.max_retries):
                delay = max(0.0, config.retry_sleep) * (2**request_attempt)
                print(
                    "[LLM retry] "
                    f"stage={stage} model_config={config.name} "
                    f"retry={request_attempt + 1}/{config.max_retries} "
                    f"wait={delay:g}s error={' '.join(last_error.split())[:500]}",
                    file=sys.stderr,
                    flush=True,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            break

        if (
            fallback_client is not None
            and content_filter_fallback_config is not None
            and is_content_moderation_error(last_error)
        ):
            metrics["content_filter_fallback_calls"] += 1
            fallback_before = llm_metrics_snapshot(fallback_client)
            print(
                "[LLM content-filter fallback] "
                f"stage={stage} primary={config.name} "
                f"fallback={content_filter_fallback_config.name}",
                file=sys.stderr,
                flush=True,
            )
            try:
                content = fallback_client(prompt, response_format_override)
            except RuntimeError as exc:
                merge_fallback_metrics(fallback_before)
                metrics["content_filter_fallback_failures"] += 1
                last_error = f"{last_error}; fallback_error={exc}"
            else:
                merge_fallback_metrics(fallback_before)
                metrics["content_filter_fallback_successes"] += 1
                metrics["successful_llm_calls"] += 1
                metrics["llm_time_seconds"] += time.perf_counter() - call_started
                return content

        metrics["failed_llm_calls"] += 1
        metrics["llm_time_seconds"] += time.perf_counter() - call_started
        raise RuntimeError(last_error)

    complete.metrics_snapshot = lambda: dict(metrics)  # type: ignore[attr-defined]
    complete.client_label = client_label or "unspecified"  # type: ignore[attr-defined]
    return complete


def extract_token_usage(body: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
        return {}
    usage = body["usage"]

    def integer(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                return int(value)
        return None

    prompt_tokens = integer("prompt_tokens", "input_tokens", "prompt_eval_count")
    completion_tokens = integer("completion_tokens", "output_tokens", "eval_count")
    total_tokens = integer("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        name: value
        for name, value in {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }.items()
        if value is not None
    }


def llm_metrics_snapshot(client: Any) -> dict[str, float | int]:
    snapshot = getattr(client, "metrics_snapshot", None)
    return dict(snapshot()) if callable(snapshot) else {}


def llm_metrics_delta(
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> dict[str, Any]:
    counters = (
        "llm_calls",
        "successful_llm_calls",
        "failed_llm_calls",
        "http_requests",
        "token_usage_reported_calls",
        "content_filter_fallback_calls",
        "content_filter_fallback_successes",
        "content_filter_fallback_failures",
    )
    result: dict[str, Any] = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in counters
    }
    result["llm_time_seconds"] = max(
        0.0,
        float(after.get("llm_time_seconds", 0.0))
        - float(before.get("llm_time_seconds", 0.0)),
    )
    for token_name, report_name in {
        "prompt_tokens": "prompt_token_reports",
        "completion_tokens": "completion_token_reports",
        "total_tokens": "total_token_reports",
    }.items():
        reports = int(after.get(report_name, 0)) - int(before.get(report_name, 0))
        result[report_name] = reports
        result[token_name] = (
            int(after.get(token_name, 0)) - int(before.get(token_name, 0))
            if reports > 0
            else None
        )
    result["token_usage_missing_calls"] = max(
        0,
        int(result["successful_llm_calls"])
        - int(result["token_usage_reported_calls"]),
    )
    return result


def combine_llm_metrics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        name: sum(int(item.get(name, 0)) for item in items)
        for name in (
            "llm_calls",
            "successful_llm_calls",
            "failed_llm_calls",
            "http_requests",
            "token_usage_reported_calls",
            "token_usage_missing_calls",
            "content_filter_fallback_calls",
            "content_filter_fallback_successes",
            "content_filter_fallback_failures",
        )
    }
    result["llm_time_seconds"] = sum(
        float(item.get("llm_time_seconds", 0.0)) for item in items
    )
    for token_name, report_name in {
        "prompt_tokens": "prompt_token_reports",
        "completion_tokens": "completion_token_reports",
        "total_tokens": "total_token_reports",
    }.items():
        result[report_name] = sum(int(item.get(report_name, 0)) for item in items)
        known = [int(item[token_name]) for item in items if item.get(token_name) is not None]
        result[token_name] = sum(known) if known else None
    return result


def select_model_config(config_path: str | Path, name: str) -> ModelApiConfig:
    configs = {config.name: config for config in load_model_api_configs(config_path)}
    try:
        return configs[name]
    except KeyError as exc:
        available = ", ".join(sorted(configs)) or "(none)"
        raise SystemExit(
            f"Unknown model config {name!r}. Available: {available}"
        ) from exc
