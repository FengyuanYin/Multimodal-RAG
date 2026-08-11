from __future__ import annotations

from agentic_rag.cli.branding import ANSI_RE, FULL_LOGO, render_startup_banner, visible_width


def test_full_color_brand_is_six_lines_and_bounded() -> None:
    banner = render_startup_banner(width=80, color=True, version="0.3.0", llm_summary="OpenAI / gpt-4.1-mini", needs_setup=False)
    lines = banner.splitlines()
    assert len(lines[:6]) == len(FULL_LOGO)
    assert "\x1b[38;2;" in banner
    assert banner.count("\x1b[0m") == 6
    assert visible_width(banner) <= 80


def test_plain_and_compact_brands_have_no_ansi() -> None:
    plain = render_startup_banner(width=80, color=False, version="0.3.0", llm_summary="LLM not configured", needs_setup=True)
    compact = render_startup_banner(width=40, color=True, version="0.3.0", llm_summary="custom?token=secret", needs_setup=False)
    assert not ANSI_RE.search(plain)
    assert "/setup" in plain
    assert ANSI_RE.sub("", compact).splitlines()[0] == "AutoMemory"
    assert "token=secret" not in compact
