from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


DEFAULT_PROMPT = "Reply with OK only."


@dataclass(slots=True)
class ModelApiConfig:
    name: str
    model: str
    base_url: str
    api_path: str
    api_key: str | None
    timeout: float
    requires_api_key: bool
    enable_thinking: bool | None = None
    max_retries: int = 3
    retry_sleep: float = 2.0
    structured_output_mode: str = "native"
    max_output_tokens_by_stage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ModelApiResult:
    name: str
    model: str
    url: str
    ok: bool
    status_code: int | None
    elapsed_seconds: float
    message: str


def load_model_api_configs(config_path: str | Path) -> list[ModelApiConfig]:
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    configs: list[ModelApiConfig] = []

    for name, value in raw.items():
        if not isinstance(value, dict):
            continue
        model = value.get("model")
        base_url = value.get("base_url")
        if not model or not base_url:
            continue

        api_path = str(value.get("api_path") or default_api_path(name, str(base_url)))
        api_key = resolve_api_key(raw, str(name))
        structured_output_mode = str(value.get("structured_output_mode", "native"))
        if structured_output_mode not in {"native", "json_object"}:
            raise ValueError(f"Invalid structured_output_mode for model config {name!r}.")
        raw_stage_limits = value.get("max_output_tokens_by_stage") or {}
        if not isinstance(raw_stage_limits, dict):
            raise ValueError(
                f"max_output_tokens_by_stage must be an object for {name!r}."
            )
        stage_limits = {
            str(stage): max(1, int(limit))
            for stage, limit in raw_stage_limits.items()
        }
        requires_api_key = bool(value.get("requires_api_key", True))
        if requires_api_key and not api_key:
            env_name = value.get("api_key_env")
            raise ValueError(
                f"Missing API key for {name!r}; set {env_name} in the environment."
                if env_name else f"Missing API key for {name!r}."
            )
        raw_enable_thinking = value.get("enable_thinking")
        enable_thinking = (
            raw_enable_thinking
            if isinstance(raw_enable_thinking, bool)
            else None
        )
        configs.append(
            ModelApiConfig(
                name=str(name),
                model=str(model),
                base_url=str(base_url),
                api_path=api_path,
                api_key=str(api_key) if api_key else None,
                timeout=float(value.get("timeout", 60)),
                requires_api_key=requires_api_key,
                enable_thinking=enable_thinking,
                max_retries=max(0, int(value.get("max_retries", 3))),
                retry_sleep=max(0.0, float(value.get("retry_sleep", 2.0))),
                structured_output_mode=structured_output_mode,
                max_output_tokens_by_stage=stage_limits,
            )
        )

    return configs


def resolve_api_key(raw: dict[str, Any], name: str) -> str | None:
    """Reuse an existing credential without duplicating it in model entries."""
    visited: set[str] = set()
    while True:
        if name in visited:
            raise ValueError(f"Circular api_key_from reference at model config {name!r}.")
        visited.add(name)
        value = raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Unknown api_key_from model config {name!r}.")
        if value.get("api_key_env"):
            return os.environ.get(str(value["api_key_env"]))
        if value.get("api_key"):
            return str(value["api_key"])
        source = value.get("api_key_from")
        if not source:
            return None
        name = str(source)


def default_api_path(name: str, base_url: str) -> str:
    lower_name = name.casefold()
    lower_url = base_url.casefold()
    if "deepseek" in lower_name or "api.deepseek.com" in lower_url:
        return "/chat/completions"
    return "/v1/chat/completions"


def chat_completion_url(config: ModelApiConfig) -> str:
    base = config.base_url.rstrip("/") + "/"
    path = config.api_path.lstrip("/")
    return urljoin(base, path)


def build_payload(
    model: str,
    prompt: str,
    token_field: str | None = "max_tokens",
    *,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    if token_field:
        payload[token_field] = 8
    return payload


def check_model_api(config: ModelApiConfig, *, prompt: str = DEFAULT_PROMPT) -> ModelApiResult:
    url = chat_completion_url(config)
    start = time.perf_counter()

    if config.requires_api_key and not config.api_key:
        return ModelApiResult(
            name=config.name,
            model=config.model,
            url=url,
            ok=False,
            status_code=None,
            elapsed_seconds=0.0,
            message="Missing api_key in config.",
        )

    attempts = ("max_tokens", "max_completion_tokens", None)
    last_error: tuple[int | None, str] | None = None
    for token_field in attempts:
        status_code, body, error = post_json(
            url,
            build_payload(
                config.model,
                prompt,
                token_field=token_field,
                enable_thinking=config.enable_thinking,
            ),
            api_key=config.api_key if config.requires_api_key else None,
            timeout=config.timeout,
        )
        elapsed = time.perf_counter() - start
        if error is None:
            return ModelApiResult(
                name=config.name,
                model=config.model,
                url=url,
                ok=True,
                status_code=status_code,
                elapsed_seconds=elapsed,
                message=extract_assistant_message(body) or "Connected.",
            )

        last_error = (status_code, error)
        if status_code != 400 or not should_retry_token_field(error):
            break

    status_code, error = last_error or (None, "Unknown error.")
    return ModelApiResult(
        name=config.name,
        model=config.model,
        url=url,
        ok=False,
        status_code=status_code,
        elapsed_seconds=time.perf_counter() - start,
        message=error,
    )


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout:
        return None, None, f"Timed out after {timeout:g}s."
    except requests.RequestException as exc:
        return None, None, f"{type(exc).__name__}: {exc}"

    text = response.text
    if not response.ok:
        return response.status_code, None, summarize_error(text)
    try:
        return response.status_code, response.json(), None
    except requests.exceptions.JSONDecodeError as exc:
        return None, None, f"Invalid JSON response: {exc}"


def summarize_error(text: str) -> str:
    text = text.strip()
    if not text:
        return "HTTP request failed with an empty response body."
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error
        return str(message)[:500]
    return json.dumps(data, ensure_ascii=False)[:500]


def should_retry_token_field(error: str) -> bool:
    lower = error.casefold()
    return "max_tokens" in lower or "max_completion_tokens" in lower


def should_retry_transient_failure(
    status_code: int | None,
    error: str | None,
) -> bool:
    """Return whether exponential-backoff retries may recover an API call."""

    if not error:
        return False
    if status_code is None:
        # post_json uses a missing status for transport failures, timeouts,
        # incomplete responses, and invalid/truncated JSON response bodies.
        return True
    if is_content_moderation_error(error):
        # Content-safety rejections are deterministic for a given input, so
        # retrying the same payload cannot succeed.
        return False
    if status_code == 400:
        # Some OpenAI-compatible gateways return a generic 400 while their
        # upstream channel is temporarily unavailable. The experiment config
        # bounds these retries, so treating it as recoverable is safe here.
        return True
    return status_code in {408, 425, 429} or 500 <= status_code <= 599


def is_content_moderation_error(error: str) -> bool:
    return "inappropriate content" in error.casefold()


def extract_assistant_message(body: dict[str, Any] | None) -> str:
    if not body:
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    text = first.get("text")
    return text.strip() if isinstance(text, str) else ""


def run_checks(
    config_path: str | Path,
    *,
    only: set[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> list[ModelApiResult]:
    configs = load_model_api_configs(config_path)
    if only:
        configs = [config for config in configs if config.name in only]
    return [check_model_api(config, prompt=prompt) for config in configs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test model API connectivity from config.json.")
    parser.add_argument("--config", default="config.json", help="Path to config.json.")
    parser.add_argument("--only", action="append", default=[], help="Config entry name to test. Can be repeated.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Short prompt used for the smoke test.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_checks(args.config, only=set(args.only) or None, prompt=args.prompt)

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            status_code = result.status_code if result.status_code is not None else "-"
            print(
                f"[{status}] {result.name} ({result.model}) "
                f"status={status_code} elapsed={result.elapsed_seconds:.2f}s"
            )
            print(f"       {result.message}")

    return 0 if results and all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
