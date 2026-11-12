#!/usr/bin/env python3
"""Literal, bounded search of the character's Markdown memory; stdlib only."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', help='Literal text, case-insensitive')
    parser.add_argument('--scope', choices=('active', 'history', 'all'), default='active')
    parser.add_argument('--limit', type=int, default=20, help='Maximum matching lines; 0 = all')
    args = parser.parse_args()
    if not args.query.strip() or args.limit < 0:
        parser.error('query must be nonempty and limit must be >= 0')
    paths = []
    if args.scope in ('active', 'all'):
        paths.extend(p for p in (ROOT / 'memory').rglob('*.md')
                     if 'archive' not in p.relative_to(ROOT / 'memory').parts)
    if args.scope in ('history', 'all'):
        paths.extend((ROOT / 'journal').rglob('*.md'))
        paths.extend((ROOT / 'memory/archive').rglob('*.md'))
    found = 0
    shown = 0
    for path in sorted(set(paths)):
        if path.is_symlink():
            continue
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if args.query.casefold() in line.casefold():
                found += 1
                if not args.limit or shown < args.limit:
                    suffix = ' [line truncated]' if len(line) > 600 else ''
                    print(f'{path.relative_to(ROOT)}:{lineno}: {line[:600]}{suffix}')
                    shown += 1
    print(f'Matches: {found}; shown: {shown}; scope: {args.scope}')
    if shown < found:
        print('Output truncated. Refine the query or use --limit 0.')


if __name__ == '__main__':
    main()
