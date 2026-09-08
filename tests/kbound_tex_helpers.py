"""Narrow live-TeX scanning helpers for manuscript regression tests."""

from __future__ import annotations

import re

_CONDITIONAL = re.compile(r"\\if[A-Za-z@]*|\\else\b|\\fi\b")


def _strip_comments(source: str) -> str:
    """Strip TeX comments while retaining percent signs escaped by an odd slash count."""
    output: list[str] = []
    for line in source.splitlines(keepends=True):
        for index, character in enumerate(line):
            if character != "%":
                continue
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                newline = "\n" if line.endswith(("\n", "\r")) else ""
                output.append(line[:index] + newline)
                break
        else:
            output.append(line)
    return "".join(output)


def live_tex(source: str) -> str:
    """Remove comments and nested ``\\iffalse`` blocks; reject malformed nesting.

    This intentionally is not a complete TeX parser. Other balanced conditionals
    remain visible because the maintained manuscript uses them for supported layout
    variants. A false block remains suppressed through nested conditionals and a
    commented terminator cannot close it.
    """
    source = _strip_comments(source)
    output: list[str] = []
    stack: list[tuple[bool, bool]] = []
    cursor = 0

    def hidden() -> bool:
        return bool(stack and stack[-1][1])

    for match in _CONDITIONAL.finditer(source):
        if not hidden():
            output.append(source[cursor : match.start()])
        token = match.group()
        if token.startswith(r"\if"):
            # `\newif\ifname` declares a conditional; it does not open one.
            if source[max(0, match.start() - 6) : match.start()] == r"\newif":
                cursor = match.end()
                continue
            parent_hidden = hidden()
            branch_hidden = parent_hidden or token == r"\iffalse"
            stack.append((parent_hidden, branch_hidden))
            if not branch_hidden:
                output.append(token)
        elif token == r"\else":
            if not stack:
                raise ValueError("unmatched TeX conditional terminator: \\else")
            if not hidden():
                output.append(token)
        else:
            if not stack:
                raise ValueError("unmatched TeX conditional terminator: \\fi")
            _, branch_hidden = stack.pop()
            if not branch_hidden and not hidden():
                output.append(token)
        cursor = match.end()

    if stack:
        raise ValueError("unterminated TeX conditional")
    output.append(source[cursor:])
    return "".join(output)
