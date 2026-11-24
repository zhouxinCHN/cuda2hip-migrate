"""Small C++ call parser used by deterministic source rules.

This is intentionally not a C++ compiler. It handles balanced function calls
while skipping comments and string/character literals, which is sufficient for
the narrow post-HIPIFY rules kept in this project.
"""

from __future__ import annotations


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def parse_call(text: str, open_index: int) -> tuple[int, list[str]] | None:
    """Return (closing_paren_index, arguments) for a balanced call."""
    depth = 1
    state = "code"
    argument_start = open_index + 1
    arguments: list[str] = []
    index = open_index + 1

    while index < len(text):
        ch = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if state == "code":
            if ch == "/" and following == "/":
                state = "line_comment"
                index += 2
                continue
            if ch == "/" and following == "*":
                state = "block_comment"
                index += 2
                continue
            if ch == '"':
                state = "string"
                index += 1
                continue
            if ch == "'":
                state = "char"
                index += 1
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    arguments.append(text[argument_start:index])
                    return index, arguments
            elif ch == "," and depth == 1:
                arguments.append(text[argument_start:index])
                argument_start = index + 1
            index += 1
            continue

        if state == "line_comment":
            if ch == "\n":
                state = "code"
            index += 1
            continue

        if state == "block_comment":
            if ch == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue

        if state in ("string", "char"):
            if ch == "\\":
                index += 2
                continue
            if (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                state = "code"
            index += 1
            continue

    return None


def find_calls(text: str, target: str) -> list[tuple[int, int, int, list[str]]]:
    """Find ``target(...)`` calls outside comments and literals.

    Returns tuples of (identifier_start, identifier_end, closing_paren, args).
    """
    calls: list[tuple[int, int, int, list[str]]] = []
    index = 0
    length = len(text)

    while index < length:
        ch = text[index]
        following = text[index + 1] if index + 1 < length else ""

        if ch == "/" and following == "/":
            newline = text.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if ch == "/" and following == "*":
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if ch in ('"', "'"):
            quote = ch
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if not _is_ident_start(ch):
            index += 1
            continue

        end = index + 1
        while end < length and _is_ident_char(text[end]):
            end += 1
        identifier = text[index:end]
        if identifier != target:
            index = end
            continue

        open_paren = end
        while open_paren < length and text[open_paren].isspace():
            open_paren += 1
        if open_paren >= length or text[open_paren] != "(":
            index = end
            continue

        parsed = parse_call(text, open_paren)
        if parsed is None:
            index = end
            continue
        close_index, arguments = parsed
        calls.append((index, end, close_index, arguments))
        index = close_index + 1

    return calls
