# knowledge/service/webapp.py
"""最小可用 Writer Web UI（stdlib `http.server`，零第三方前端框架）。

运行方式（见 README / DEVLOG）：
    python -m knowledge.service.webapp [--port 8765]

只做 UI 层：收集表单 → `writer.build_request` → `writer.generate` → 渲染结果。
业务逻辑全部在 `knowledge/service/writer.py`（共享服务层）；本文件绝不重实现
StylePlanner / PromptCompiler / Generation / Evaluation / Revision / Feedback Loop，
也绝不读取 / 打印 / 暴露 DEEPSEEK_API_KEY。

- 作者下拉框来自 `writer.list_authors()`（Generic Author Registry），非硬编码；
  not-ready 作者显示 "Not ready — author profile has not been built" 且不可生成。
- Generate = 真实 LLM 调用（DeepSeek）；feedback 单轮为额外 API 消耗（可选）。
- 生成正文仅存会话内存（`_RESULTS`），绝不落盘 / 绝不提交 Git；下载为 .txt。
"""
from __future__ import annotations

import html
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .writer import WriterError, build_request, generate, list_authors

_DEFAULT_PORT = 8765

# 会话内结果缓存（keyed by generation_id），下载 .txt 用；进程退出即消失。
_RESULTS: dict[str, dict[str, Any]] = {}

_DESIRED_LENGTHS = ("short_scene", "scene", "excerpt", "chapter")


def _escape(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _form_page(authors: list[dict[str, Any]], *, error: str | None = None) -> str:
    options: list[str] = []
    for a in authors:
        label = _escape(a["display_name"])
        if a["ready"]:
            options.append(f'<option value="{_escape(a["author_id"])}">{label}</option>')
        else:
            # not-ready 作者：禁用 + 明确提示（绝不硬编码，来自 registry）。
            options.append(
                f'<option value="{_escape(a["author_id"])}" disabled>'
                f'{label} — Not ready — author profile has not been built</option>')
    if not options:
        options.append(
            '<option value="" disabled>No ready authors — '
            'build an author profile first (see onboarding)</option>')

    length_options = "\n".join(
        f'<option value="{d}"{" selected" if d == "short_scene" else ""}>{d}</option>'
        for d in _DESIRED_LENGTHS)

    err_html = f'<div class="error">{_escape(error)}</div>' if error else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Weaver Style Engine — Writer</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 label {{ display: block; margin-top: .9rem; font-weight: 600; }}
 textarea, select, input[type=number], input[type=text] {{ width: 100%; margin-top: .25rem; font-size: 1rem; }}
 textarea {{ min-height: 7rem; }}
 .hint {{ color: #666; font-size: .85rem; font-weight: 400; }}
 .error {{ color: #b00020; background: #fdecea; padding: .6rem .8rem; border-radius: 4px; margin: 1rem 0; }}
 button {{ margin-top: 1.2rem; padding: .55rem 1.1rem; font-size: 1rem; cursor: pointer; }}
</style></head><body>
<h1>Weaver Style Engine — Writer</h1>
<p class="hint">Generate a style-conditioned passage using a <em>real</em> LLM call
(DeepSeek). The author dropdown comes from the Generic Author Registry — only authors
whose profile has been built can generate. Requires <code>DEEPSEEK_API_KEY</code>.</p>
{err_html}
<form method="post" action="/generate">
  <label>Author
    <select name="author">{"".join(options)}</select>
  </label>
  <label>Writing request (scene brief)
    <textarea name="content" required placeholder="Describe the scene to write…"></textarea>
  </label>
  <label>Desired length
    <select name="desired_length">{length_options}</select>
  </label>
  <label>Target words <span class="hint">(optional)</span>
    <input type="number" name="target_words" min="1" placeholder="e.g. 400">
  </label>
  <label>Point of view <span class="hint">(optional, e.g. third / first)</span>
    <input type="text" name="pov" placeholder="third">
  </label>
  <label>Constraints <span class="hint">(optional, one per line)</span>
    <textarea name="constraints" placeholder="Do not introduce new named characters."></textarea>
  </label>
  <label><input type="checkbox" name="feedback" value="1">
    Evaluate &amp; optimize (one feedback pass — extra API cost)</label>
  <button type="submit">Generate</button>
</form>
</body></html>"""


def _result_page(author_id: str, display_name: str, r: dict[str, Any]) -> str:
    fb = r.get("feedback")
    feedback_html = ""
    if fb:
        if fb.get("status") == "failed":
            # 反馈优化失败：初稿已成功生成，明确警示，绝不伪装成成功/优化结果。
            feedback_html = (
                f'<p class="warn">初稿已生成，但自动评价/优化失败：'
                f'{_escape(fb.get("reason", ""))}</p>')
        else:
            feedback_html = (
                f'<p>Feedback (1 pass): outcome <code>{_escape(fb.get("outcome"))}</code>'
                f' — {_escape((fb.get("decision") or {}).get("reason", ""))}</p>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Generated — {_escape(display_name)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 1rem; border-radius: 6px; line-height: 1.5; }}
 .meta {{ color: #555; font-size: .9rem; }}
 .warn {{ color: #8a6d00; background: #fff8e1; padding: .6rem .8rem; border-radius: 4px; margin: 1rem 0; }}
 a {{ display: inline-block; margin-top: 1rem; }}
</style></head><body>
<h1>Generated passage — {_escape(display_name)}</h1>
<p class="meta">author <code>{_escape(author_id)}</code> · {_escape(r.get("word_count"))} words ·
finish <code>{_escape(r.get("finish_reason"))}</code> · provider/model
<code>{_escape(r.get("provider"))}</code>/<code>{_escape(r.get("model"))}</code></p>
{feedback_html}
<a href="/download?generation_id={_escape(r.get("generation_id"))}">Download as .txt</a>
<pre>{_escape(r.get("generated_text"))}</pre>
<p><a href="/">&larr; New request</a></p>
</body></html>"""


class WriterRequestHandler(BaseHTTPRequestHandler):
    """极简路由：GET /（表单）· POST /generate · GET /download。"""

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, *, filename: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header(
            "Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_form_page(list_authors()))
            return
        if parsed.path == "/download":
            gid = (parse_qs(parsed.query).get("generation_id") or [""])[0]
            rec = _RESULTS.get(gid)
            if rec is None:
                self._send_html(_form_page(list_authors(),
                                           error="结果已失效，请重新生成。"), 404)
                return
            filename = f"{rec['author_id']}_writer.txt"
            self._send_text(rec["text"], filename=filename)
            return
        self._send_html(_form_page(list_authors(), error="未知路径。"), 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/generate":
            self._send_html(_form_page(list_authors(), error="未知路径。"), 404)
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        form = {k: (v[0] if v else "") for k, v in parse_qs(raw).items()}

        author_id = (form.get("author") or "").strip()
        content = (form.get("content") or "").strip()
        desired_length = (form.get("desired_length") or "short_scene").strip()
        pov = (form.get("pov") or "").strip() or None
        feedback = int(form.get("feedback") == "1")

        target_words: int | None = None
        if (form.get("target_words") or "").strip():
            try:
                target_words = int(form["target_words"].strip())
            except ValueError:
                self._send_html(_form_page(list_authors(),
                                           error="target_words 必须是整数。"), 400)
                return

        constraints = [
            ln.strip() for ln in (form.get("constraints") or "").splitlines()
            if ln.strip()]

        if not content:
            self._send_html(_form_page(list_authors(), error="写作需求不能为空。"), 400)
            return

        try:
            request = build_request(
                content, desired_length=desired_length, target_words=target_words,
                language="english", pov=pov, constraints=constraints)
            result = generate(author_id, request, feedback_iterations=feedback)
        except WriterError as e:
            self._send_html(_form_page(list_authors(), error=str(e)), 400)
            return
        except Exception as e:  # noqa: BLE001 — 兜底，绝不向用户泄露内部堆栈/密钥
            # 浏览器侧只显示异常类型（不泄露内部堆栈/密钥）；终端侧留完整 traceback，
            # 便于定位真实根因（真实人工验收时正是这里吞掉了 traceback）。
            traceback.print_exc()
            self._send_html(_form_page(
                list_authors(), error=f"生成失败：{type(e).__name__}"), 500)
            return

        _RESULTS[result["generation_id"]] = {
            "author_id": result["author_id"],
            "text": result["generated_text"],
        }
        self._send_html(_result_page(
            result["author_id"], result["display_name"], result))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = _DEFAULT_PORT
    if argv:
        if argv[0] == "--port" and len(argv) == 2:
            port = int(argv[1])
        else:
            print("用法: python -m knowledge.service.webapp [--port 8765]",
                  file=sys.stderr)
            return 2

    server = ThreadingHTTPServer(("127.0.0.1", port), WriterRequestHandler)
    print(f"Weaver Style Engine — Writer UI 已启动： http://127.0.0.1:{port}/")
    print("（Generate = 真实 LLM 调用；需 DEEPSEEK_API_KEY；Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
