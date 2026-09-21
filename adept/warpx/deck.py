"""WarpX (AMReX ParmParse) inputs parser / renderer.

WarpX reads a flat, non-sectioned key/value namespace::

    # comment
    max_step = 100
    amr.n_cell = 128
    geometry.prob_lo = 0.       # meters — WarpX decks are SI
    electrons.density_function(x,y,z) = "n0*exp(z/Ln)"
    diag1.intervals = 100:200:10

Facts about the format this module relies on:

- Keys are dotted names; parser-function keys carry an argument list verbatim
  (``density_function(x,y,z)``) which is part of the key string.
- Values are whitespace-separated tokens; a value list runs until the next
  ``key =`` definition, so it may span lines (AMReX tokenizes, it does not
  parse line-by-line).
- ``#`` starts a comment outside quotes.
- Strings with spaces (parser expressions) are quoted with ``"`` or ``'``.
- A repeated key overrides the earlier definition (last one wins), matching
  ParmParse query semantics.

Canonical representation: an insertion-ordered ``dict`` mapping the full key
string (including any ``(...)`` argument spec) to a Python value — int, float,
str, or a list of those. No bool conversion: ParmParse itself stores strings
and 0/1 ints, so ``true``/``false`` stay strings.

The invariant we rely on is dict-level round-trip:

    parse(render(parse(text))) == parse(text)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

Deck = dict[str, Any]

_QUOTE_CHARS = "\"'"

# A ParmParse key: dotted identifier, optionally followed by a parser-argument
# list that is kept verbatim as part of the key, e.g. "density_function(x,y,z)".
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.@\-]*(?:\([^)]*\))?")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    """Drop everything from the first unquoted ``#`` to the end of line."""
    quote = ""
    out: list[str] = []
    for ch in line:
        if quote:
            if ch == quote:
                quote = ""
            out.append(ch)
        elif ch in _QUOTE_CHARS:
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _tokenize(src: str) -> list[tuple[str, bool]]:
    """Split into ``(text, was_quoted)`` tokens on unquoted whitespace.

    A quoted region keeps its content (quotes stripped) as a single token even
    if it contains spaces; adjacent unquoted text sticks to it, so
    ``pre"a b"post`` is one token — good enough, decks don't do that.
    """
    tokens: list[tuple[str, bool]] = []
    buf: list[str] = []
    quoted = False
    quote = ""
    for ch in src:
        if quote:
            if ch == quote:
                quote = ""
            else:
                buf.append(ch)
        elif ch in _QUOTE_CHARS:
            quote = ch
            quoted = True
        elif ch in " \t\n\r":
            if buf or quoted:
                tokens.append(("".join(buf), quoted))
            buf, quoted = [], False
        else:
            buf.append(ch)
    if quote:
        raise ValueError("Unterminated quote in WarpX inputs")
    if buf or quoted:
        tokens.append(("".join(buf), quoted))
    return tokens


def _split_glued_eq(tokens: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Split unquoted ``key=value`` tokens into ``key``, ``=``, ``value``.

    Only splits when the part before the first ``=`` is a complete ParmParse
    key and the character after it is not another ``=`` (so a parser expression
    like ``x==0`` is left alone; expressions containing a single ``=`` must be
    quoted, which the renderer guarantees).
    """
    out: list[tuple[str, bool]] = []
    for text, quoted in tokens:
        eq = text.find("=")
        if not quoted and eq == 0 and len(text) > 1 and text[1] != "=":
            # "=0.9" — the value glued to a standalone "=".
            out.append(("=", False))
            out.extend(_split_glued_eq([(text[1:], False)]))
            continue
        if quoted or eq <= 0 or text[eq : eq + 2] == "==":
            out.append((text, quoted))
            continue
        head = text[:eq]
        m = _KEY_RE.match(head)
        if m and m.end() == len(head):
            out.append((head, False))
            out.append(("=", False))
            rest = text[eq + 1 :]
            if rest:
                out.extend(_split_glued_eq([(rest, False)]))
        else:
            out.append((text, quoted))
    return out


def _parse_atom(text: str, quoted: bool) -> Any:
    if quoted:
        return text
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def parse_deck(text: str) -> Deck:
    """Parse WarpX inputs text into an ordered flat dict."""
    src = "\n".join(_strip_comment(ln) for ln in text.splitlines())
    tokens = _split_glued_eq(_tokenize(src))

    # A token starts a new definition iff the following token is "=".
    def is_key_at(i: int) -> bool:
        if i + 1 >= len(tokens):
            return False
        text, quoted = tokens[i]
        nxt, nxt_quoted = tokens[i + 1]
        if quoted or nxt_quoted or nxt != "=":
            return False
        m = _KEY_RE.match(text)
        return m is not None and m.end() == len(text)

    deck: Deck = {}
    i = 0
    while i < len(tokens):
        if not is_key_at(i):
            raise ValueError(f"Expected 'key =' at token {tokens[i][0]!r}")
        key = tokens[i][0]
        i += 2  # skip key and "="
        values: list[Any] = []
        while i < len(tokens) and not is_key_at(i):
            text, quoted = tokens[i]
            if text == "=" and not quoted:
                raise ValueError(f"Stray '=' in value of {key!r}")
            values.append(_parse_atom(text, quoted))
            i += 1
        if not values:
            raise ValueError(f"Key {key!r} has no value")
        # Repeated key: last definition wins, but keep first-seen position.
        deck[key] = values[0] if len(values) == 1 else values
    return deck


def parse_deck_file(path: str | Path) -> Deck:
    return parse_deck(Path(path).read_text())


def _needs_quotes(s: str) -> bool:
    if s == "":
        return True
    if any(ch in s for ch in " \t\n#=\"'"):
        return True
    # A bare token that would re-parse as a number must be quoted to survive
    # the round trip as a string.
    return bool(_INT_RE.match(s) or _FLOAT_RE.match(s))


def _render_atom(v: Any) -> str:
    if isinstance(v, bool):  # from YAML overrides; ParmParse reads 0/1
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"' if _needs_quotes(v) else v
    raise TypeError(f"Cannot render value of type {type(v).__name__}: {v!r}")


def _render_value(v: Any) -> str:
    if isinstance(v, list):
        return " ".join(_render_atom(x) for x in v)
    return _render_atom(v)


def render_deck(deck: Deck) -> str:
    """Render the flat dict back to WarpX inputs text, one key per line."""
    kw = max((len(k) for k in deck), default=0)
    lines = [f"{k:<{kw}} = {_render_value(v)}" for k, v in deck.items()]
    return "\n".join(lines) + "\n"


def _find_deck_key(deck: Deck, requested: str) -> str:
    """Resolve an override key against the deck's existing keys.

    Accepts either an exact key (``"electrons.density_function(x,y,z)"``) or
    the base name without the argument spec (``"electrons.density_function"``),
    the latter only when unambiguous. Returns the requested string verbatim if
    the key is brand-new.
    """
    if requested in deck:
        return requested
    candidates = [k for k in deck if k.split("(", 1)[0] == requested]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return requested
    raise ValueError(f"Ambiguous override key {requested!r}; candidates: {candidates}")


def merge_overrides(deck: Deck, overrides: dict[str, Any]) -> None:
    """Apply ``overrides`` (flat ``{"amr.n_cell": 512, ...}``) in place.

    Unknown keys are appended — WarpX ignores unused parameters, and a new key
    (e.g. an extra reduced diagnostic parameter) is a legitimate override.

    Values are TYPED, not deck text: a multi-valued ParmParse parameter must be
    a list (``"warpx.numprocs": [2, 16]`` renders as ``= 2 16``). The string
    ``"2 16"`` renders *quoted* (strings with whitespace must be re-quoted to
    round-trip) and makes WarpX abort at startup in ``ReadParameters``.
    """
    for k, v in (overrides or {}).items():
        deck[_find_deck_key(deck, k)] = v


_KEY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_./:\- ]+")


def _sanitize_key(k: str) -> str:
    """Map a deck key (which may contain ``(x,y,z)``) to an MLflow-safe name."""
    out = _KEY_SANITIZE_RE.sub("_", k)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "_"


def deck_to_flat_dict(deck: Deck) -> dict[str, Any]:
    """Flat, MLflow-safe representation for param logging.

    ``electrons.density_function(x,y,z)`` → ``electrons.density_function_x_y_z``;
    list values get numeric suffixes (``amr.n_cell.0``, ...).
    """
    flat: dict[str, Any] = {}
    for k, v in deck.items():
        ks = _sanitize_key(k)
        if isinstance(v, list):
            for j, x in enumerate(v):
                flat[f"{ks}.{j}"] = x
        else:
            flat[ks] = v
    return flat
