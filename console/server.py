"""crew console — локальный веб-чат с ИИ-сотрудниками команды.

Запуск: uv run server.py [--host 127.0.0.1] [--port 8766]

Ключевая идея: запуск агента — объект на сервере, а не HTTP-запрос. Браузер только
читает append-only журнал событий с нужного индекса. Отсюда переживание рефреша,
закрытой крышки и подключения с телефона к тому же разговору.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import signal
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CREW = HERE.parent  # корень репозитория: рядом лежат каталоги crew_<имя>

# Каждому агенту — свой способ вызова. Добавить сотрудника = одна строка.
# `cwd` — домашний каталог сотрудника, `notes_dir` — папка заметок внутри memory/,
# `she` — род для строк вида «Лора сейчас занята».
AGENTS: dict[str, dict[str, Any]] = {
    "lora": {
        "name": "Лора",
        "role": "ML / Data Science",
        "kind": "claude",
        "cwd": str(CREW / "crew_lora"),
        "she": True,
    },
    "steve": {
        "name": "Стив",
        "role": "Python / бекенд",
        "kind": "claude",
        "cwd": str(CREW / "crew_steve"),
    },
    "alfred": {
        "name": "Альфред",
        "role": "доступы, ключи, заявки",
        "kind": "claude",
        "cwd": str(CREW / "crew_alfred"),
        # Его единица знания — не проект, а система, к которой есть доступ.
        "notes_dir": "access",
        "notes_word": "системах",
    },
}

CLAUDE = shutil.which("claude") or "claude"

# Кто сидит по эту сторону консоли. Приложение однопользовательское: имя уезжает
# агентам в системный промпт, чтобы они отличали человека от скрипта.
CALLER = os.environ.get("CREW_CALLER", "создатель")

# Сотрудникам это уезжает в системный промпт при каждом запуске. Главное здесь —
# третий пункт: в ленте видны только названия инструментов, но не их вывод, и без
# этой оговорки агент отвечает «смотри выше», а смотреть человеку некуда.
CLAUDE_PREAMBLE = f"""\
Ты отвечаешь в crew console — локальном веб-чате. С тобой говорит человек \
({CALLER}), а не другой агент и не скрипт.

Что из этого следует:
- Это прямая работа, а не консультация у соседа: правила про «советуй, но не правь» \
здесь не действуют, поручения выполняй сам.
- Человек видит твой текст и названия вызванных инструментов, но НЕ видит их вывод. \
Всё существенное, что нашёл в файлах или в выводе команд, перескажи словами.
- Ответ отображается как markdown.
- Диалог продолжается между сообщениями: прошлые реплики помнишь, переспросить можно.
- Отвечай кратко: это чат, а не отчёт.
- Зовёшь соседа — это отдельный агент, он идёт минутами. Проси у него короткий ответ \
по существу, а не исследование: узкий вопрос закрывается за минуту, широкий не \
укладывается и в десять.
- Обычный вызов ставь с `timeout: 600000` — это потолок Bash. Утилиты `timeout` на \
macOS нет, оборачивать в неё команду не надо.
- Не уложился — перезапусти с `run_in_background: true`. Придёт путь к файлу вывода. \
`claude -p` пишет в него не постепенно, а один раз в конце, поэтому пустой файл \
означает «ещё думает», а не «завис». Дождись ответа until-циклом и только потом читай:
  `until [ -s <файл> ]; do sleep 10; done`
  Голый `sleep` для ожидания харнесс блокирует — нужен именно until-цикл."""

# Дельты копим и сбрасываем пачкой: иначе строка в JSONL на каждый токен.
DELTA_FLUSH_CHARS = 300
DELTA_FLUSH_SECONDS = 0.12

STATE: dict[str, dict[str, Any]] = {}
TOKEN: str | None = None


# ──────────────────────────────────────────────────────────── журнал событий


# Заголовок заметки: `name: harness-bench-fast — бенчмарк агентных харнессов`.
NOTE_NAME = re.compile(r"^name:\s*(.+)$", re.M)
LABEL_MAX = 26


def projects(agent: str) -> list[dict[str, str]]:
    """Проекты сотрудника — по одному файлу заметки в его памяти.

    Список читается на каждый запрос: заметки заводятся и переименовываются, а
    listdir дешевле, чем объяснять, почему новый проект не появился.

    `id` — имя файла (по нему агент найдёт заметку), `label` — человеческое имя
    из фронтматтера: там сохранён регистр (`LoRA`, `PyTorch`), которого в именах
    файлов нет.
    """
    cfg = AGENTS[agent]
    if cfg.get("projects") is not None:
        return [{"id": p, "label": p} for p in cfg["projects"]]

    src = Path(cfg["cwd"]) / "memory" / cfg.get("notes_dir", "projects")
    if not src.is_dir():
        return []

    out = []
    for path in sorted(src.glob("*.md")):
        if path.stem.lower() == "readme":
            continue
        label = path.stem
        try:
            head = path.read_text(encoding="utf-8")[:400]
        except OSError:
            head = ""
        m = NOTE_NAME.search(head)
        if m:
            # До тире — само название, после — пояснение, оно на кнопке лишнее.
            name = re.split(r"\s+[—–-]\s+", m.group(1).strip(), maxsplit=1)[0].strip()
            if name and len(name) <= LABEL_MAX:
                label = name
        out.append({"id": path.stem, "label": label})
    return out


def log_path(agent: str) -> Path:
    return DATA / f"{agent}.jsonl"


def emit(agent: str, **ev: Any) -> dict[str, Any]:
    """Добавить событие в журнал агента и разбудить всех, кто его читает."""
    st = STATE[agent]
    ev["i"] = len(st["events"])
    st["events"].append(ev)
    with log_path(agent).open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    st["wake"].set()
    st["wake"] = asyncio.Event()
    return ev


def load_log(agent: str) -> None:
    """Поднять журнал с диска: экран не пустой после перезапуска сервера."""
    st = STATE[agent]
    path = log_path(agent)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        ev["i"] = len(st["events"])
        st["events"].append(ev)
        if ev.get("t") == "session":
            st["session"] = ev.get("id")
        elif ev.get("t") == "reset":
            st["session"] = None


def reap(agent: str) -> None:
    """Запуски, не пережившие перезапуск сервера, закрыть честной пометкой.

    Автоматически не переспрашиваем: молча повторить агента с полными правами
    хуже, чем показать, что запуск оборвался.
    """
    st = STATE[agent]
    open_run = False
    for ev in st["events"]:
        t = ev.get("t")
        if t == "start":
            open_run = True
        elif t in ("done", "error", "stopped"):
            open_run = False
    if open_run:
        emit(agent, t="stopped", note="сервер был перезапущен, запуск оборвался")


# ──────────────────────────────────────────────────────────── бекенд claude


def tool_brief(name: str, args: dict[str, Any]) -> str:
    """Короткая подпись тул-вызова для ленты."""
    for key in ("file_path", "path", "pattern", "command", "url", "prompt", "query"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            one = " ".join(val.split())
            return one[:120] + ("…" if len(one) > 120 else "")
    return ""


# Харнессы, которыми сотрудник поднимает соседа. Требуем пробел после имени:
# иначе имя проекта вроде `deepagents-something` в аргументах grep выглядит вызовом.
HARNESS = re.compile(r"\b(claude|codex|free-code|deepagents|kimi)\s")
# И вопрос должен куда-то передаваться — без этого команда просто упоминает имя.
PROMPTING = re.compile(r"(?:^|\s)(?:-p|--print|-z|--oneshot|exec)(?:\s|$)")

# Вопрос доезжает до соседа тремя разными путями, и их все надо уметь прочитать.
Q_FLAG = re.compile(r"""(?:-p|--print|-z|--oneshot|exec)\s+(['"])(.+?)\1""", re.S)
Q_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\s*\1", re.S)
Q_PIPE = re.compile(
    r"""\b(?:printf|echo)\s+(?:-\w+\s+)*(?:['"]%[sb]['"]\s+)?(['"])(.+?)\1\s*(?:\||$)""",
    re.S,
)


def extract_question(command: str) -> str:
    """Достать текст вопроса из команды вызова соседа.

    Порядок от самого надёжного к самому общему: heredoc, конвейер через
    printf/echo (так вопрос уходит харнессам, которые читают его из stdin), и
    наконец просто аргумент флага.
    """
    m = Q_HEREDOC.search(command)
    if m:
        return m.group(2).strip()

    m = Q_PIPE.search(command)
    if m and m.group(2).strip() not in ("%s", "%b"):
        return m.group(2).strip()

    m = Q_FLAG.search(command)
    if m and "$(cat)" not in m.group(2):
        return m.group(2).strip()

    return ""


def detect_consult(command: str, caller: str) -> tuple[str, str] | None:
    """Понять, что этой Bash-командой зовут соседа: (кого, о чём).

    Сотрудник консультируется, запуская харнесс в чужом каталоге. В ленте это
    приходит обычным Bash-вызовом, по которому видно только слово «Bash», —
    его-то и хочется развернуть в отдельный блок с аватаром вызванного.
    """
    # Оба признака обязательны: харнесс именно запускается, и ему передают
    # вопрос. Одного упоминания имени мало — `grep` по памяти соседки это
    # чтение, а не обращение к ней.
    if not (HARNESS.search(command) and PROMPTING.search(command)):
        return None

    # Себя спросить нельзя: путь собственного дома сплошь и рядом попадает в
    # текст вопроса («Лора спрашивает, из /.../crew_lora»), и по простому
    # совпадению подстроки агент оказывался собеседником сам себе.
    others = {k: c for k, c in AGENTS.items() if k != caller}

    target = None
    # Надёжный признак — в чей каталог заходят, а не любое упоминание пути.
    for key, cfg in others.items():
        if cfg.get("cwd") and re.search(rf"\bcd\s+{re.escape(cfg['cwd'])}\b", command):
            target = key
            break
    if target is None:  # последняя попытка: просто упоминание чужого дома
        for key, cfg in others.items():
            if cfg.get("cwd") and cfg["cwd"] in command:
                target = key
                break
    if target is None:
        return None
    return target, extract_question(command)


# Консультация дольше 10 минут упирается в потолок Bash, и агент перезапускает
# её в фоне. Тогда вместо ответа приходит расписка — выдавать её за ответ нельзя.
# Из расписки забираем путь к файлу вывода: сосед пишет в него по мере работы,
# и по росту файла видно, что он жив, а не завис.
BACKGROUND = re.compile(
    r"running in background with ID:?\s*([\w-]+).*?written to:?\s*(\S+)", re.I | re.S
)
# Read отдаёт файл в формате `cat -n` — номера строк для чата лишние.
LINENO = re.compile(r"^\s*\d+\t", re.M)


def tool_result_text(block: dict[str, Any]) -> str:
    """Достать текст из tool_result: content бывает и строкой, и списком блоков."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


async def watch_background(agent: str, who: str, path: str) -> None:
    """Следить за файлом вывода фоновой консультации.

    Спрашивающий опрашивает файл редко, и между его опросами блок в ленте
    выглядел бы замершим. Пока файл растёт — сосед работает, и это видно.
    """
    started = time.monotonic()
    last = -1
    try:
        while True:
            await asyncio.sleep(2)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            # Пустой файл — это норма: `claude -p` отдаёт ответ одним куском в
            # конце, поэтому «ничего не написал» не значит «завис». Показываем
            # время ожидания, а как только пошёл текст — его объём.
            if size != last or size == 0:
                last = size
                emit(
                    agent,
                    t="consult_progress",
                    who=who,
                    size=size,
                    waited=int(time.monotonic() - started),
                )
    except asyncio.CancelledError:
        pass


async def run_claude(agent: str, text: str) -> None:
    """Лора и Стив: claude с рабочим каталогом в доме сотрудника."""
    cfg = AGENTS[agent]
    st = STATE[agent]
    sid = st["session"]

    args = [
        CLAUDE,
        "-p",
        text,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--permission-mode",
        "bypassPermissions",
        # Кто зовёт и куда придёт ответ — в системный промпт, а не в текст вопроса:
        # так это не мусорит в ленте и не путается с самим вопросом.
        "--append-system-prompt",
        CLAUDE_PREAMBLE,
    ]
    args += ["--resume", sid] if sid else ["--session-id", str(uuid4())]

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cfg["cwd"],
        # Без DEVNULL claude ждёт stdin три секунды на каждом запуске.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Дефолтный лимит 64 КиБ — первая же большая строка tool_result роняет
        # запуск на LimitOverrunError в середине.
        limit=16 * 1024 * 1024,
        # Своя группа процессов: без неё Stop убьёт claude, а его детей
        # (pytest, npm) оставит сиротами.
        start_new_session=True,
    )
    st["proc"] = proc

    buf: list[str] = []
    last_flush = time.monotonic()
    consults: dict[str, str] = {}  # id вызова тула → кого спросили
    background: dict[str, str] = {}  # id фоновой задачи и путь вывода → кого спросили
    watchers: list[asyncio.Task] = []  # слежение за фоновыми консультациями

    def flush() -> None:
        nonlocal last_flush
        if buf:
            emit(agent, t="delta", text="".join(buf))
            buf.clear()
        last_flush = time.monotonic()

    try:
        async for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue  # одна битая строка не должна ронять пятиминутный запуск

            kind = msg.get("type")

            if kind == "system" and msg.get("subtype") == "init":
                # apiKeySource == "none" значит вход по подписке, а не по API-ключу:
                # тогда total_cost_usd — пересчёт токенов по тарифам API, справочный,
                # деньги по нему не списываются. Помечаем, чтобы UI не врал.
                st["billed"] = msg.get("apiKeySource") not in (None, "none")
                new_sid = msg.get("session_id")
                # resume может форкнуть сессию — всегда берём id из init,
                # иначе непрерывность тихо ломается со второго запуска.
                if new_sid and new_sid != st["session"]:
                    st["session"] = new_sid
                    emit(agent, t="session", id=new_sid)

            elif kind == "stream_event":
                delta = msg.get("event", {}).get("delta", {})
                # Только text_delta: thinking_delta и signature_delta — это
                # размышления и base64-подпись, им в чате не место.
                if delta.get("type") == "text_delta":
                    buf.append(delta.get("text", ""))
                    if (
                        sum(len(x) for x in buf) >= DELTA_FLUSH_CHARS
                        or time.monotonic() - last_flush >= DELTA_FLUSH_SECONDS
                    ):
                        flush()

            elif kind == "assistant":
                # Текстовые блоки здесь дублируют text_delta — берём только тулы.
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") != "tool_use":
                        continue
                    flush()
                    name = block.get("name", "?")
                    args = block.get("input") or {}
                    # Ответ на фоновую консультацию агент забирает, читая файл
                    # её вывода. Ловим Read именно по этому пути.
                    if name == "Read" and str(args.get("file_path", "")) in background:
                        consults[block.get("id")] = background[str(args["file_path"])]
                        continue
                    if name == "BashOutput":  # запасной путь для других харнессов
                        bg_id = str(args.get("bash_id") or args.get("shell_id") or "")
                        if bg_id in background:
                            consults[block.get("id")] = background[bg_id]
                            continue

                    consult = (
                        detect_consult(str(args.get("command", "")), agent)
                        if name == "Bash"
                        else None
                    )
                    if consult:
                        who, question = consult
                        consults[block.get("id")] = who
                        # Команду кладём всегда: если вопрос вытащить не вышло
                        # (например, его записали во временный файл), пусть в
                        # ленте будет видно хотя бы её, а не пустой блок.
                        emit(
                            agent,
                            t="consult",
                            who=who,
                            question=question,
                            cmd="" if question else str(args.get("command", ""))[:1500],
                        )
                    else:
                        emit(agent, t="tool", name=name, brief=tool_brief(name, args))

            elif kind == "user":
                # Результаты тулов не показываем — кроме ответа вызванного соседа.
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    who = consults.pop(block.get("tool_use_id"), None)
                    if not who:
                        continue
                    answer = tool_result_text(block).strip()
                    bg = BACKGROUND.search(answer)
                    if bg:
                        # Не ответ, а расписка. Запоминаем и id, и путь к выводу,
                        # и начинаем следить за файлом: пока он растёт — сосед жив.
                        bg_id, path = bg.group(1), bg.group(2).rstrip(".,")
                        background[bg_id] = who
                        background[path] = who
                        emit(agent, t="consult_bg", who=who)
                        watchers.append(
                            asyncio.create_task(watch_background(agent, who, path))
                        )
                    else:
                        emit(
                            agent,
                            t="consult_reply",
                            who=who,
                            answer=LINENO.sub("", answer)[:20000],
                            failed=bool(block.get("is_error")),
                        )

            elif kind == "result":
                flush()
                if st["stopping"]:
                    # После Stop claude отдаёт result с error_during_execution —
                    # это и есть наш стоп, а не сбой. Закроет run_agent.
                    pass
                elif msg.get("is_error"):
                    emit(agent, t="error", text=str(msg.get("result") or msg.get("subtype")))
                else:
                    usage = msg.get("usage") or {}
                    emit(
                        agent,
                        t="done",
                        cost=msg.get("total_cost_usd"),
                        billed=st.get("billed", False),
                        ms=msg.get("duration_ms"),
                        turns=msg.get("num_turns"),
                        out_tokens=usage.get("output_tokens"),
                    )

        flush()
        code = await proc.wait()
        # Ненулевой код после Stop — это и есть Stop, а не ошибка.
        if code != 0 and not ended(agent) and not st["stopping"]:
            err = (await proc.stderr.read()).decode(errors="replace").strip()
            emit(agent, t="error", text=err[-1500:] or f"claude завершился с кодом {code}")
    finally:
        for w in watchers:
            w.cancel()
        st["proc"] = None


def ended(agent: str) -> bool:
    """Уже закрыли текущий запуск финальным событием?"""
    for ev in reversed(STATE[agent]["events"]):
        t = ev.get("t")
        if t in ("done", "error", "stopped"):
            return True
        if t == "start":
            return False
    return False


# ──────────────────────────────────────────────────────────── оркестрация


async def run_agent(
    agent: str,
    text: str,
    chosen: list[str] | None = None,
    labels: list[str] | None = None,
) -> None:
    st = STATE[agent]
    st["stopping"] = False
    emit(agent, t="user", text=text, projects=labels or chosen or [], ts=time.time())
    emit(agent, t="start", ts=time.time())
    if chosen:
        # Даём и человеческое имя, и файл заметки: по имени понятно, о чём речь,
        # по файлу — что открыть.
        sub = AGENTS[agent].get("notes_dir", "projects")
        names = ", ".join(
            f"{lab} (memory/{sub}/{pid}.md)"
            for pid, lab in zip(chosen, labels or chosen)
        )
        text = (
            f"[Речь идёт о {AGENTS[agent].get('notes_word', 'проектах')}: {names}. "
            f"Загляни в свои заметки по ним, прежде чем отвечать.]\n\n{text}"
        )
    try:
        await run_claude(agent, text)
    except Exception as exc:  # упавший запуск не должен уносить сервер
        if not ended(agent):
            emit(agent, t="error", text=f"{type(exc).__name__}: {exc}")
    finally:
        st["task"] = None
        if not ended(agent):
            emit(agent, t="stopped")
        st["stopping"] = False


def busy_text(agent: str) -> str:
    cfg = AGENTS[agent]
    return f"{cfg['name']} сейчас {'занята' if cfg.get('she') else 'занят'}"


def is_running(agent: str) -> bool:
    task = STATE[agent]["task"]
    return task is not None and not task.done()


async def stop_agent(agent: str) -> None:
    """SIGTERM всей группе процессов, через 3 секунды SIGKILL.

    Задачу не отменяем: убитый процесс сам закроет поток, и run_agent доиграет
    до конца штатно. Так меньше гонок, чем с CancelledError на середине разбора.
    """
    st = STATE[agent]
    st["stopping"] = True
    proc = st["proc"]
    if proc is None or proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(30):
            await asyncio.sleep(0.1)
            if proc.returncode is not None:
                return
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ──────────────────────────────────────────────────────────── HTTP


app = FastAPI(title="crew console")


@app.middleware("http")
async def auth(request: Request, call_next):
    """Единственная защита: токен, если сервер открыт наружу.

    Полные права + доступ по сети = выполнение чего угодно для любого, кто
    откроет страницу. На 127.0.0.1 токен не нужен и не спрашивается.
    """
    if TOKEN is None:
        return await call_next(request)
    given = request.query_params.get("t")
    if given and secrets.compare_digest(given, TOKEN):
        response = (
            RedirectResponse("/") if request.url.path == "/" else await call_next(request)
        )
        response.set_cookie("crew_token", TOKEN, httponly=True, samesite="lax")
        return response
    cookie = request.cookies.get("crew_token")
    if cookie and secrets.compare_digest(cookie, TOKEN):
        return await call_next(request)
    return JSONResponse({"error": "нужен токен: откройте /?t=<токен>"}, status_code=401)


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.get("/avatars/{name}")
async def avatar(name: str):
    # Имя берём из ключей реестра, а не из запроса, — обхода каталога быть не может.
    if name.removesuffix(".jpg") not in AGENTS:
        return JSONResponse({"error": "нет такого сотрудника"}, status_code=404)
    path = HERE / "avatars" / f"{name.removesuffix('.jpg')}.jpg"
    if not path.exists():
        return JSONResponse({"error": "аватар не найден"}, status_code=404)
    return FileResponse(path)


@app.get("/api/agents")
async def api_agents():
    return [
        {
            "id": key,
            "name": cfg["name"],
            "role": cfg["role"],
            "kind": cfg["kind"],
            "she": bool(cfg.get("she")),
            "running": is_running(key),
            "projects": projects(key),
            "events": len(STATE[key]["events"]),
        }
        for key, cfg in AGENTS.items()
    ]


@app.post("/api/send")
async def api_send(request: Request):
    body = await request.json()
    agent = body.get("agent")
    text = (body.get("text") or "").strip()
    if agent not in AGENTS:
        return JSONResponse({"error": "нет такого сотрудника"}, status_code=404)
    if not text:
        return JSONResponse({"error": "пустой вопрос"}, status_code=400)
    if is_running(agent):
        return JSONResponse(
            {"error": busy_text(agent)}, status_code=409
        )
    # Отмеченные проекты подставляются в начало вопроса, но в ленте остаются
    # отдельной пометкой: в пузыре человека — то, что он набрал.
    known = {p["id"]: p["label"] for p in projects(agent)}
    chosen = [p for p in (body.get("projects") or []) if p in known]
    labels = [known[p] for p in chosen]
    STATE[agent]["task"] = asyncio.create_task(run_agent(agent, text, chosen, labels))
    return JSONResponse({"ok": True}, status_code=202)


@app.post("/api/stop")
async def api_stop(request: Request):
    body = await request.json()
    agent = body.get("agent")
    if agent not in AGENTS:
        return JSONResponse({"error": "нет такого сотрудника"}, status_code=404)
    await stop_agent(agent)
    return {"ok": True}


@app.post("/api/reset")
async def api_reset(request: Request):
    """Новый тред: своя сессия и чистый экран. Журнал на диске переписывается."""
    body = await request.json()
    agent = body.get("agent")
    if agent not in AGENTS:
        return JSONResponse({"error": "нет такого сотрудника"}, status_code=404)
    if is_running(agent):
        return JSONResponse(
            {"error": busy_text(agent)}, status_code=409
        )
    st = STATE[agent]
    st["events"] = []
    st["session"] = None
    log_path(agent).write_text("", encoding="utf-8")
    emit(agent, t="reset")
    return {"ok": True}


@app.get("/api/events/{agent}")
async def api_events(agent: str, request: Request):
    if agent not in AGENTS:
        return JSONResponse({"error": "нет такого сотрудника"}, status_code=404)
    start = request.query_params.get("from")
    last_id = request.headers.get("last-event-id")
    try:
        cursor = int(last_id) + 1 if last_id else int(start or 0)
    except ValueError:
        cursor = 0

    async def gen():
        i = cursor
        while True:
            st = STATE[agent]
            # Пробуждение берём ДО вычитывания: событие, пришедшее пока мы
            # отдаём накопленное, дёрнет именно этот Event, и мы не уснём на
            # 15 секунд с непрочитанным хвостом.
            wake = st["wake"]
            while i < len(st["events"]):
                ev = st["events"][i]
                yield f"id: {ev['i']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                i += 1
            try:
                await asyncio.wait_for(wake.wait(), timeout=15)
            except asyncio.TimeoutError:
                # Пинг доказывает браузеру и прокси, что труба жива.
                yield ": ping\n\n"
            if await request.is_disconnected():
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────── запуск


def main() -> None:
    global TOKEN

    parser = argparse.ArgumentParser(description="crew console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    opts = parser.parse_args()

    if opts.host not in ("127.0.0.1", "localhost"):
        TOKEN = os.environ.get("CREW_CONSOLE_TOKEN")
        if not TOKEN:
            raise SystemExit(
                "Отказ: сервер открыт наружу без токена.\n"
                "У сотрудников полные права — любой в этой сети получил бы агента,\n"
                "которому можно всё. Задай токен и повтори:\n\n"
                f'  CREW_CONSOLE_TOKEN=$(openssl rand -hex 16) uv run server.py --host {opts.host}\n\n'
                "Телефон подключается один раз по ссылке /?t=<токен>."
            )

    DATA.mkdir(exist_ok=True)
    for agent in AGENTS:
        STATE[agent] = {
            "events": [],
            "session": None,
            "proc": None,
            "task": None,
            "stopping": False,
            "wake": asyncio.Event(),
        }
        load_log(agent)
        reap(agent)

    import uvicorn

    print(f"crew console → http://{opts.host}:{opts.port}")
    if TOKEN:
        print(f"  телефон: http://<ip-этой-машины>:{opts.port}/?t={TOKEN}")
    # Без --reload намеренно: сохранение файла убивало бы живых агентов.
    uvicorn.run(app, host=opts.host, port=opts.port, log_level="warning")


if __name__ == "__main__":
    main()
