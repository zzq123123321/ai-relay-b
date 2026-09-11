"""集中式错误翻译函数的行为测试。

覆盖：上游模型/OpenCode 固定错误、OpenChamber 连接与完成判定、
Reasonix、剪贴板协议、中文透传、幂等与兜底。
"""

from __future__ import annotations

from core.error_translator import translate_error


def test_sse_read_timeout_translated_with_original():
    out = translate_error("SSE read timed out")
    assert "流式响应读取超时" in out
    assert "原始错误：SSE read timed out" in out


def test_connection_reset_translated_with_original():
    out = translate_error("Connection reset by server")
    assert "主动断开连接" in out
    assert "原始错误：Connection reset by server" in out


def test_bad_request_maps_to_model_rejection():
    out = translate_error("Bad Request")
    assert "模型拒绝了当前请求" in out
    assert "原始错误：Bad Request" in out


def test_opencode_failed_to_send_message():
    out = translate_error("OpenCode failed to send a message 400")
    assert "OpenCode 未能发送消息" in out
    assert "原始错误：" in out


def test_structured_qwen_rejection_maps_to_model_rejection():
    out = translate_error(
        "APIError\nstatusCode: 400\nisRetryable: false\n"
        "url: http://192.168.100.190:8080/v1/chat/completions"
    )
    assert "模型拒绝了当前请求" in out
    assert "原始错误：APIError" in out


def test_timeout_keeps_seconds():
    out = translate_error(
        "OpenChamber task did not finish within 900s; session "
        "ses_a may still be running; open it in OpenChamber and check the result"
    )
    assert "900 秒内未完成" in out
    assert "ses_a" in out
    assert "原始错误：" in out


def test_cannot_reach_translated():
    out = translate_error("cannot reach OpenChamber at http://127.0.0.1:57123")
    assert "无法连接 OpenChamber 服务" in out
    assert "127.0.0.1" in out


def test_auth_error_translated():
    out = translate_error(
        "OpenChamber rejected the request (HTTP 401); configure normal "
        "OpenChamber UI authentication instead of disabling it"
    )
    assert "认证失败" in out


def test_truncation_translated():
    out = translate_error(
        "OpenChamber reply was truncated by the model output limit "
        "(session ses_x); this is not a success"
    )
    assert "被模型输出长度上限截断" in out


def test_ambiguous_translated():
    out = translate_error(
        "ambiguous: this task round cannot be uniquely attributed in "
        "OpenChamber session ses_x (multiple new user messages or a "
        "parentID mismatch; a message may have been typed manually)"
    )
    assert "无法唯一归属" in out


def test_abort_translated():
    out = translate_error(
        "OpenChamber task failed in session ses_x: Aborted"
    )
    assert "失败" in out
    assert "ses_x" in out


def test_task_failed_with_marker_preserves_session():
    out = translate_error("OpenChamber task failed in session ses_x: Aborted")
    assert "ses_x" in out


def test_reasonix_window_not_found_translated():
    out = translate_error("Reasonix window was not found")
    assert "未找到 Reasonix 窗口" in out


def test_protocol_marker_translated():
    out = translate_error("missing AI_RELAY/1 marker")
    assert "缺少 AI_RELAY/1 标记" in out


def test_relay_prefix_stripped_before_matching():
    out = translate_error(
        "openchamber_execute:OpenChamberTimeoutError: did not finish within 30s"
    )
    assert "30 秒内未完成" in out
    assert "原始错误：openchamber_execute:" in out


def test_chinese_input_passthrough():
    text = "OpenChamber 项目目录未配置，请在设置中填写"
    assert translate_error(text) == text


def test_already_translated_is_idempotent():
    once = translate_error("SSE read timed out")
    assert translate_error(once) == once
    once_cn = translate_error("OpenChamber 项目目录未配置")
    assert translate_error(once_cn) == once_cn


def test_unknown_english_gets_chinese_wrapper():
    out = translate_error("no endpoint")
    assert "发生错误" in out
    assert "原始错误：no endpoint" in out


def test_empty_input_returns_empty():
    assert translate_error("") == ""
    assert translate_error(None) is None or translate_error(None) == ""


def test_mixed_english_with_chinese_timeout_note():
    out = translate_error(
        "OpenChamber task did not finish within 200s; session ses_x may "
        "still be running；若会话正等待你的问题/权限确认，仍可在 "
        "OpenChamber 中处理; open it in OpenChamber and check the result"
    )
    assert "200 秒内未完成" in out
    assert "等待你的问题" in out