"""OSIRIS namelist parser / renderer.

OSIRIS uses a Fortran-namelist-like syntax that is NOT a standard Fortran
namelist:

    ! comment
    section_name
    {
      key            = scalar,
      key(1:N)       = a, b, c,    ! comments allowed inline
      key            = "string",
      key            = .true.,
      key(1:2,1)     = 0.0, 1.0,
    }

Multiple sections with the same name are common (e.g. ``species``) and the
order matters because it corresponds to species 1, 2, ... in OSIRIS.

Canonical representation: an ordered list of ``(section_name, params)``
pairs where ``params`` is a ``dict`` mapping a key-string (with any slice
spec preserved verbatim, e.g. ``"nx_p(1:1)"``) to a Python value (int,
float, str, bool, or list of those).

The invariant we rely on is dict-level round-trip:

    parse(render(parse(text))) == parse(text)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

Section = tuple[str, dict[str, Any]]
Sections = list[Section]


_TRUE_TOKENS = {".true.", ".t."}
_FALSE_TOKENS = {".false.", ".f."}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    """Drop everything from the first unquoted ``!`` to the end of line."""
    in_string = False
    out: list[str] = []
    for ch in line:
        if ch == '"':
            in_string = not in_string
            out.append(ch)
        elif ch == "!" and not in_string:
            break
        else:
            out.append(ch)
    return "".join(out)


def _split_top_commas(s: str) -> list[str]:
    """Split on commas not inside double-quotes."""
    pieces: list[str] = []
    buf: list[str] = []
    in_string = False
    for ch in s:
        if ch == '"':
            in_string = not in_string
            buf.append(ch)
        elif ch == "," and not in_string:
            pieces.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    last = "".join(buf).strip()
    if last:
        pieces.append(last)
    return pieces


def _parse_atom(tok: str) -> Any:
    """Convert a single token to a Python value."""
    t = tok.strip()
    if not t:
        return None
    if t.startswith('"') and t.endswith('"') and len(t) >= 2:
        return t[1:-1]
    low = t.lower()
    if low in _TRUE_TOKENS:
        return True
    if low in _FALSE_TOKENS:
        return False
    if _INT_RE.match(t):
        return int(t)
    if _FLOAT_RE.match(t):
        return float(t.replace("d", "e").replace("D", "e"))
    return t  # unrecognized — keep as-is


def _parse_value(rhs: str) -> Any:
    pieces = _split_top_commas(rhs)
    if not pieces:
        return None
    if len(pieces) == 1:
        return _parse_atom(pieces[0])
    return [_parse_atom(p) for p in pieces]


def parse_deck(text: str) -> Sections:
    """Parse an OSIRIS native deck (as a string) into ordered sections."""
    src = "\n".join(_strip_comment(ln) for ln in text.splitlines())
    n = len(src)
    i = 0

    def skip_ws(j: int) -> int:
        while j < n and src[j] in " \t\n\r":
            j += 1
        return j

    sections: Sections = []
    while True:
        i = skip_ws(i)
        if i >= n:
            break
        m = _IDENT_RE.match(src, i)
        if not m:
            raise ValueError(f"Expected section name at offset {i}: {src[i : i + 30]!r}")
        name = m.group(0)
        i = m.end()
        i = skip_ws(i)
        if i >= n or src[i] != "{":
            raise ValueError(f"Expected '{{' after section {name!r} at offset {i}: {src[i : i + 30]!r}")
        i += 1
        params: dict[str, Any] = {}
        while True:
            i = skip_ws(i)
            if i >= n:
                raise ValueError(f"Unterminated section {name!r}")
            if src[i] == "}":
                i += 1
                break
            km = _KEY_RE.match(src, i)
            if not km:
                raise ValueError(f"Expected key in section {name!r} at offset {i}: {src[i : i + 30]!r}")
            key = km.group(0)
            i = km.end()
            i = skip_ws(i)
            if i >= n or src[i] != "=":
                raise ValueError(f"Expected '=' after key {key!r} at offset {i}")
            i += 1
            value_start = i
            in_string = False
            while i < n:
                ch = src[i]
                if ch == '"':
                    in_string = not in_string
                    i += 1
                elif not in_string and ch == "}":
                    break
                elif not in_string and ch == ",":
                    # Peek past whitespace: if the next token is an identifier
                    # followed by '=', this comma terminates the current
                    # key=value pair; otherwise it's an intra-value separator
                    # (e.g. ``uth(1:3) = 0.1, 0.2, 0.3``).
                    j = i + 1
                    while j < n and src[j] in " \t\n\r":
                        j += 1
                    peek = _KEY_RE.match(src, j)
                    if peek:
                        jj = peek.end()
                        while jj < n and src[jj] in " \t\n\r":
                            jj += 1
                        if jj < n and src[jj] == "=":
                            break
                    i += 1
                else:
                    i += 1
            value_text = src[value_start:i].strip()
            if i < n and src[i] == ",":
                i += 1
            params[key] = _parse_value(value_text)
        sections.append((name, params))
    return sections


def parse_deck_file(path: str | Path) -> Sections:
    return parse_deck(Path(path).read_text())


def _render_value(v: Any) -> str:
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return ", ".join(_render_value(x) for x in v)
    if v is None:
        return ""
    raise TypeError(f"Cannot render value of type {type(v).__name__}: {v!r}")


def render_deck(sections: Sections) -> str:
    """Render canonical sections back to OSIRIS native namelist text."""
    out: list[str] = []
    for name, params in sections:
        out.append(name)
        if not params:
            out.append("{")
            out.append("}")
            out.append("")
            continue
        out.append("{")
        kw = max(len(k) for k in params)
        for k, v in params.items():
            out.append(f"  {k:<{kw}} = {_render_value(v)},")
        out.append("}")
        out.append("")
    return "\n".join(out)


def _find_param_key(params: dict[str, Any], requested: str) -> str:
    """Resolve an override key against a section's existing keys.

    Accepts either an exact key (``"nx_p(1:1)"``) or the base name
    (``"nx_p"``), the latter only when it's unambiguous within the section.
    Returns the matching existing key, or the requested string verbatim if
    the key is brand-new.
    """
    if requested in params:
        return requested
    candidates = [k for k in params if k.split("(", 1)[0] == requested]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return requested
    raise ValueError(f"Ambiguous override key {requested!r}; candidates: {candidates}")


def _merge_params(params: dict[str, Any], over: dict[str, Any]) -> None:
    for k, v in over.items():
        target = _find_param_key(params, k)
        params[target] = v


def merge_overrides(sections: Sections, overrides: dict[str, Any]) -> None:
    """Apply ``overrides`` to ``sections`` in place.

    ``overrides`` shape::

        {
            "grid": {"nx_p": [256]},  # apply to all occurrences
            "species": {0: {"num_par_x": [512]}},  # indexed for repeated sections
        }
    """
    if not overrides:
        return
    by_name: dict[str, list[int]] = {}
    for idx, (name, _) in enumerate(sections):
        by_name.setdefault(name, []).append(idx)

    for sec_name, sec_over in overrides.items():
        if sec_name not in by_name:
            raise KeyError(f"Override references unknown section: {sec_name!r}")
        occurrences = by_name[sec_name]
        if isinstance(sec_over, dict) and sec_over and all(isinstance(k, int) for k in sec_over.keys()):
            for idx, params_over in sec_over.items():
                if idx < 0 or idx >= len(occurrences):
                    raise IndexError(
                        f"Override section {sec_name!r}[{idx}] out of range; {len(occurrences)} occurrence(s) present"
                    )
                _merge_params(sections[occurrences[idx]][1], params_over)
        else:
            for occ in occurrences:
                _merge_params(sections[occ][1], sec_over)


_KEY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_./:\- ]+")


def _sanitize_key(k: str) -> str:
    """Map an OSIRIS key (which may contain ``()``, ``[]``) to an MLflow-safe
    parameter name. MLflow allows alphanumerics, ``_``, ``-``, ``.``, ``:``,
    ``/``, and spaces. Everything else is folded to ``_`` and trailing
    runs of underscores are collapsed."""
    out = _KEY_SANITIZE_RE.sub("_", k)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "_"


def deck_to_flat_dict(sections: Sections) -> dict[str, Any]:
    """Flat representation for MLflow param logging.

    Repeated sections get bracketed indices in the source, then sanitized
    to MLflow-safe names: ``species[0].num_par_x(1:1)`` →
    ``species_0.num_par_x_1:1``. List values get numeric suffixes.
    """
    name_count: dict[str, int] = {}
    for name, _ in sections:
        name_count[name] = name_count.get(name, 0) + 1
    name_idx: dict[str, int] = {}
    flat: dict[str, Any] = {}
    for name, params in sections:
        if name_count[name] > 1:
            i = name_idx.get(name, 0)
            prefix = _sanitize_key(f"{name}[{i}]")
            name_idx[name] = i + 1
        else:
            prefix = _sanitize_key(name)
        for k, v in params.items():
            ks = _sanitize_key(k)
            if isinstance(v, list):
                for j, x in enumerate(v):
                    flat[f"{prefix}.{ks}.{j}"] = x
            else:
                flat[f"{prefix}.{ks}"] = v
    return flat
