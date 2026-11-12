#!/usr/bin/env python3
"""Validate local links, card metadata, index reachability and startup size."""
import datetime
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TYPES = {'feedback': 'feedback', 'lessons': 'lesson', 'projects': 'project', 'reference': 'reference'}
LINK = re.compile(r'\[[^\]\n]*\]\(([^)\n]+)\)')


def targets(path, text):
    # Ignore fenced examples. External URLs and anchors require a separate check.
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    for target in LINK.findall(text):
        target = target.strip().strip('<>')
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        yield (path.parent / unquote(parsed.path)).resolve()


def main():
    errors = []
    warnings = []
    graph = {}
    paths = sorted(ROOT.rglob('*.md'))
    for path in paths:
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_relative_to(ROOT):
                errors.append(f'{rel}: broken or external symlink')
            continue
        text = path.read_text(encoding='utf-8')
        links = set(targets(path, text))
        graph[path.resolve()] = links
        for target in links:
            if not target.is_relative_to(ROOT):
                errors.append(f'{rel}: link outside package')
            elif not target.exists():
                errors.append(f'{rel}: missing link {target.relative_to(ROOT)}')
        parts = rel.parts
        card_type = None
        if rel.as_posix() == 'memory/stack.md':
            card_type = 'reference'
        elif len(parts) >= 3 and parts[0] == 'memory' and parts[1] in TYPES and path.name != 'README.md':
            card_type = TYPES[parts[1]]
            index = path.parent / 'README.md'
            if not index.exists() or path.resolve() not in set(targets(index, index.read_text(encoding='utf-8'))):
                errors.append(f'{rel}: missing from section README')
        if card_type:
            match = re.match(r'\A---\n(.*?)\n---(?:\n|$)', text, re.S)
            if not match:
                errors.append(f'{rel}: missing frontmatter')
                continue
            meta = {}
            for line in match[1].splitlines():
                key, sep, value = line.partition(':')
                if not sep or key in meta:
                    errors.append(f'{rel}: invalid or duplicate metadata field')
                meta[key] = value.strip()
            if set(meta) != {'name', 'description', 'type', 'updated'} or not all(meta.values()):
                errors.append(f'{rel}: requires nonempty name, description, type, updated')
            if meta.get('name') != path.stem or meta.get('type') != card_type:
                errors.append(f'{rel}: name/type mismatch')
            try:
                value = meta.get('updated', '')
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                    raise ValueError
                datetime.date.fromisoformat(value)
            except ValueError:
                errors.append(f'{rel}: invalid updated date')
        if parts[0] == 'memory' and path.stat().st_size > 15 * 1024:
            warnings.append(f'{rel}: >15 KiB; review manually')
    for name in ('AGENTS.md', 'memory/MEMORY.md', 'CLAUDE.md'):
        if not (ROOT / name).exists():
            errors.append(f'{name}: missing entry point')
    if (ROOT / 'CLAUDE.md').exists() and (ROOT / 'CLAUDE.md').resolve() != (ROOT / 'AGENTS.md').resolve():
        errors.append('CLAUDE.md must be a symlink to AGENTS.md')
    entry = ROOT / 'memory/MEMORY.md'
    seen = set()
    pending = [entry.resolve()]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, set()) - seen)
    for path in paths:
        rel = path.relative_to(ROOT)
        if rel.parts[0] == 'memory' and 'archive' not in rel.parts and path.resolve() not in seen:
            errors.append(f'{rel}: unreachable from memory map')
    startup = sum((ROOT / name).stat().st_size for name in ('AGENTS.md', 'memory/MEMORY.md') if (ROOT / name).exists())
    if startup > 12 * 1024:
        warnings.append(f'startup {startup} bytes >12 KiB')
    for issue in errors:
        print('ERROR:', issue)
    for issue in warnings:
        print('WARNING:', issue)
    print(f'{len(errors)} errors, {len(warnings)} warnings; startup {startup} bytes')
    print('Only local structure checked; remote state and factual correctness not checked.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
