"""集中式错误翻译：把底层英文错误文本转成面向操作者的中文提示。

报错原因必须先以中文呈现给用户，英文原文保留在“原始错误：”一行
用于排查。relay / UI 本身已经以中文为主的提示原样返回，避免重复包装。
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ORIGINAL_MARKER = "原始错误："

_RELAY_PREFIX_RE = re.compile(
    r"^(?:openchamber_execute|openchamber_cancel|openchamber_config|"
    r"reasonix_execute|error|invalid clipboard task)\s*:\s*",
    re.IGNORECASE,
)


def _has_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def _strip_relay_prefix(text: str) -> str:
    return _RELAY_PREFIX_RE.sub("", text.strip())


_TIMEOUT_RULE = re.compile(r"did not finish within (\d+)s")


_RULES: list[tuple[re.Pattern[str], str]] = [
    # ---- 上游模型 / OpenCode 固定错误（OpenChamber 侧系统提示） ---- #
    (
        re.compile(r"SSE read timed out", re.IGNORECASE),
        "模型流式响应读取超时，正在重试。",
    ),
    (
        re.compile(r"connection reset by server|connection reset", re.IGNORECASE),
        "模型服务器主动断开连接，正在重试。",
    ),
    (
        re.compile(r"OpenCode failed to send a message", re.IGNORECASE),
        "OpenCode 未能发送消息，请稍后重试。",
    ),
    (
        re.compile(r"model rejected the request|rejected the request \(HTTP 400\)"),
        "模型拒绝了当前请求，可能是会话上下文过长。",
    ),
    (
        re.compile(r"isRetryable: false"),
        "模型拒绝了当前请求（服务声明不可重试），可能是会话上下文过长。",
    ),
    (
        re.compile(r"^bad request$", re.IGNORECASE),
        "模型拒绝了当前请求，可能是会话上下文过长。",
    ),
    # ---- OpenChamber 连接与配置 ---- #
    (
        re.compile(r"cannot reach OpenChamber at (\S+)"),
        "无法连接 OpenChamber 服务（{0}）。",
    ),
    (
        re.compile(r"connection refused|connection error|cannot reach", re.IGNORECASE),
        "无法连接 OpenChamber 服务。",
    ),
    (
        re.compile(r"request.*timed out|timed out", re.IGNORECASE),
        "请求超时。",
    ),
    (
        re.compile(
            r"configure normal OpenChamber UI authentication|authentication|"
            r"HTTP 401|HTTP 403|unauthorized",
            re.IGNORECASE,
        ),
        "OpenChamber 认证失败，请检查凭据配置。",
    ),
    (
        re.compile(r"OpenChamber address must not be empty"),
        "OpenChamber 服务地址未配置。",
    ),
    (
        re.compile(r"reports unhealthy status"),
        "OpenChamber 服务处于不健康状态。",
    ),
    (
        re.compile(r"non-JSON body|invalid payload|not a list|has an invalid payload"),
        "OpenChamber 服务返回的数据格式异常。",
    ),
    (
        re.compile(r"session create response has no sessionId"),
        "创建会话失败：服务未返回会话 ID。",
    ),
    (
        re.compile(r"returned HTTP \d{3}"),
        "OpenChamber 服务返回了错误状态码。",
    ),
    # ---- 执行与完成判定 ---- #
    (
        _TIMEOUT_RULE,
        "OpenChamber 任务在 {0} 秒内未完成，可能仍在执行或等待人工确认。",
    ),
    (
        re.compile(r"ambiguous"),
        "OpenChamber 本轮结果无法唯一归属（存在多条新消息、父消息不符或时间戳无法排序），未作猜测判定。",
    ),
    (
        re.compile(r"truncated by the model output limit"),
        "OpenChamber 回复被模型输出长度上限截断，未视为成功。",
    ),
    (
        re.compile(r"task failed in session (\S+)"),
        "OpenChamber 会话 {0} 中的任务失败。",
    ),
    (
        re.compile(r"\bAborted\b"),
        "OpenChamber 任务被中止（Aborted）。",
    ),
    (
        re.compile(r"the last assistant message of this round is not completed"),
        "OpenChamber 本轮最后的助手回复尚未完成。",
    ),
    (
        re.compile(r"has no assistant reply for this task"),
        "OpenChamber 会话没有本轮任务的助手回复，任务可能尚未开始执行。",
    ),
    (
        re.compile(r"was never recorded in session"),
        "OpenChamber 任务未在会话中被记录，请检查会话。",
    ),
    (
        re.compile(r"did not dispatch the prompt"),
        "OpenChamber 未接受并转交该任务，任务未发送，会话保留用于手动检查。",
    ),
    (
        re.compile(r"finish=\s*None|ended abnormally"),
        "OpenChamber 本轮异常结束（缺少结束标记）。",
    ),
    (
        re.compile(r"round ended abnormally"),
        "OpenChamber 本轮异常结束。",
    ),
    (
        re.compile(r"cannot attribute this round"),
        "无法归属本轮任务结果（缺少可验证的用户消息）。",
    ),
    (
        re.compile(r"model must be formatted as providerID/modelID"),
        "模型格式无效，应为“providerID/modelID”。",
    ),
    (
        re.compile(r"task prompt must not be empty"),
        "任务内容不能为空。",
    ),
    (
        re.compile(r"failed to open session deep link"),
        "请求打开 OpenChamber 会话失败（系统未能处理会话链接）。",
    ),
    (
        re.compile(r"task was already processed"),
        "该任务之前已被处理，拒绝重复执行。",
    ),
    # ---- Reasonix ---- #
    (
        re.compile(r"Reasonix window was not found", re.IGNORECASE),
        "未找到 Reasonix 窗口（可能未打开或窗口标题已变化）。",
    ),
    (
        re.compile(r"Reasonix composer input was not found", re.IGNORECASE),
        "未找到 Reasonix 输入框。",
    ),
    (
        re.compile(r"Reasonix send button (was not found|is disabled)", re.IGNORECASE),
        "未找到 Reasonix 发送按钮或按钮不可用。",
    ),
    (
        re.compile(r"Reasonix control was not found: (\S+)", re.IGNORECASE),
        "未找到 Reasonix 控件（{0}）。",
    ),
    (
        re.compile(r"Reasonix did not enter generation state", re.IGNORECASE),
        "Reasonix 未进入生成状态。",
    ),
    (
        re.compile(r"timed out waiting for Reasonix response", re.IGNORECASE),
        "等待 Reasonix 回复超时。",
    ),
    (
        re.compile(r"Reasonix reply boundary was not found", re.IGNORECASE),
        "未找到 Reasonix 回复结束标记。",
    ),
    (
        re.compile(r"Reasonix send invocation failed", re.IGNORECASE),
        "Reasonix 发送调用多次失败。",
    ),
    (
        re.compile(r"Reasonix task must not be empty", re.IGNORECASE),
        "Reasonix 任务内容不能为空。",
    ),
    (
        re.compile(r"comtypes is required", re.IGNORECASE),
        "缺少 comtypes/uiautomation 依赖。",
    ),
    # ---- AI Relay 剪贴板协议 ---- #
    (
        re.compile(r"missing AI_RELAY/1 marker"),
        "剪贴板消息缺少 AI_RELAY/1 标记。",
    ),
    (
        re.compile(r"missing blank line before message body"),
        "剪贴板消息正文前缺少空行。",
    ),
    (
        re.compile(r"duplicate protocol header"),
        "协议头重复。",
    ),
    (
        re.compile(r"invalid protocol header"),
        "协议头格式无效。",
    ),
    (
        re.compile(r"missing protocol headers"),
        "缺少必需协议头。",
    ),
    (
        re.compile(r"missing protocol header"),
        "缺少协议头。",
    ),
    (
        re.compile(r"message body must not be empty"),
        "消息正文不能为空。",
    ),
    (
        re.compile(r"unsupported message type"),
        "不支持的消息类型。",
    ),
    (
        re.compile(r"response body must not be empty"),
        "响应正文不能为空。",
    ),
    (
        re.compile(r"in_reply_to must not be empty|IN_REPLY_TO must not be empty"),
        "IN_REPLY_TO 不能为空。",
    ),
    (
        re.compile(r"invalid ROUND/MAX_ROUNDS"),
        "ROUND 或 MAX_ROUNDS 无效。",
    ),
    (
        re.compile(r"ROUND and MAX_ROUNDS must be integers"),
        "ROUND 和 MAX_ROUNDS 必须是整数。",
    ),
    (
        re.compile(r"missing AI_RELAY_END marker"),
        "缺少 AI_RELAY_END 结束标记。",
    ),
    # ---- 应用自身 —— #
    (
        re.compile(r"AI Relay startup import failed"),
        "AI Relay 启动时导入依赖失败。",
    ),
    (
        re.compile(r"AI Relay B端已在运行|已在运行"),
        "AI Relay 已在运行。",
    ),
]


def _wants_timeout_note(text: str, template: str) -> str:
    """Append a waiting-on-operator hint for a timeout when OPENCHAMBER
    already embedded a (Chinese) pending-user-action note in the payload."""
    if "等待你的问题" in text or "权限确认" in text:
        return f"{template}（检测到会话正等待你的问题/权限确认）"
    return template


def translate_error(text: str) -> str:
    """把一段错误文本翻译成面向操作者的中文提示。

    - 以中文为主的文本原样返回（避免把中文提示里的英文尾部误翻）；
    - 命中已知模式时返回中文提示，并在“原始错误：”一行附上英文原文；
    - 未命中的纯英文文本也会包一层中文引导，避免界面出现大段英文。
    """
    if not text or not text.strip():
        return text or ""
    stripped = text.strip()
    if _ORIGINAL_MARKER in stripped:
        return stripped
    if _cjk_ratio(stripped) >= 0.3:
        return stripped

    probe = _strip_relay_prefix(stripped)
    for pattern, template in _RULES:
        match = pattern.search(probe)
        if not match:
            continue
        reason = template
        if "{" in reason:
            reason = reason.format(*match.groups())
        if pattern == _TIMEOUT_RULE:
            reason = _wants_timeout_note(stripped, reason)
        return f"{reason}\n{_ORIGINAL_MARKER}{stripped}"

    reason = _short_fallback(probe)
    return f"{reason}\n{_ORIGINAL_MARKER}{stripped}"


def _short_fallback(probe: str) -> str:
    first_line = probe.splitlines()[0].strip()
    if len(first_line) > 40:
        first_line = f"{first_line[:40]}…"
    return f"发生错误（{first_line}）。"