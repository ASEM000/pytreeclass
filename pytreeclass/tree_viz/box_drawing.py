from __future__ import annotations

from typing import Any, Sequence

import jax.tree_util as jtu


def _hbox(*text):

    boxes = list(map(_vbox, text))
    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    FMT = ""

    for _, line in enumerate(zip(*boxes)):
        FMT += _resolve_line(line) + "\n"
    return FMT


def _hstack(boxes):
    boxes = [(box).split("\n") for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    # expand height of each col before merging
    boxes = [b + [" " * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    FMT = ""

    _cells = tuple(zip(*boxes))

    for i, line in enumerate(_cells):
        FMT += _resolve_line(line) + ("\n" if i != (len(_cells) - 1) else "")

    return FMT


def _vbox(*text: tuple[str, ...]) -> str:
    """Create vertically stacked text boxes

    Returns:
        str: stacked boxes string

    Examples:
        >>> _vbox("a","b")
        ┌───┐
        │a  │
        ├───┤
        │b  │
        └───┘

        >>> _vbox("a","","a")
        ┌───┐
        │a  │
        ├───┤
        │   │
        ├───┤
        │a  │
        └───┘
    """

    max_width = (
        max(jtu.tree_leaves([[len(t) for t in item.split("\n")] for item in text])) + 0
    )

    top = f"┌{'─'*max_width}┐"
    line = f"├{'─'*max_width}┤"
    side = [
        "\n".join([f"│{t}{' '*(max_width-len(t))}│" for t in item.split("\n")])
        for item in text
    ]
    btm = f"└{'─'*max_width}┘"

    formatted = ""

    for i, s in enumerate(side):

        if i == 0:
            formatted += f"{top}\n{s}\n{line if len(side)>1 else btm}"

        elif i == len(side) - 1:
            formatted += f"\n{s}\n{btm}"

        else:
            formatted += f"\n{s}\n{line}"

    return formatted


def _resolve_line(cols: Sequence[str, ...]) -> str:
    """combine columns of single line by merging their borders

    Args:
        cols (Sequence[str,...]): Sequence of single line column string

    Returns:
        str: resolved column string

    Example:
        >>> _resolve_line(['ab','b│','│c'])
        'abb│c'

        >>> _resolve_line(['ab','b┐','┌c'])
        'abb┬c'

    """

    cols = list(map(list, cols))  # convert each col to col of chars
    alpha = ["│", "┌", "┐", "└", "┘", "┤", "├"]

    for index in range(len(cols) - 1):

        if cols[index][-1] == "┐" and cols[index + 1][0] in ["┌", "─"]:
            cols[index][-1] = "┬"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "┘" and cols[index + 1][0] in ["└", "─"]:
            cols[index][-1] = "┴"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "┤" and cols[index + 1][0] in ["├", "─", "└"]:  #
            cols[index][-1] = "┼"
            cols[index + 1].pop(0)

        elif cols[index][-1] in ["┘", "┐", "─"] and cols[index + 1][0] in ["├"]:
            cols[index][-1] = "┼"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "─" and cols[index + 1][0] == "└":
            cols[index][-1] = "┴"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "─" and cols[index + 1][0] == "┌":
            cols[index][-1] = "┬"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "│" and cols[index + 1][0] == "─":
            cols[index][-1] = "├"
            cols[index + 1].pop(0)

        elif cols[index][-1] == " ":
            cols[index].pop()

        elif cols[index][-1] in alpha and cols[index + 1][0] in [*alpha, " "]:
            cols[index + 1].pop(0)

    return "".join(map(lambda x: "".join(x), cols))


def _table(lines: Sequence[str, ...]) -> str:
    """create a table with self aligning rows and cols

    Args:
        lines (Sequence[str,...]): list of lists of cols values

    Returns:
        str: box string

    Example:
        >>> print(_table([['1\n','2'],['3','4000']]))
        ┌─┬────────┐
        │1│3       │
        │ │        │
        ├─┼────────┤
        │2│40000000│
        └─┴────────┘
    """
    # align _cells vertically
    for i, _cells in enumerate(zip(*lines)):
        max_cell_height = max(map(lambda x: x.count("\n"), _cells))
        for j in range(len(_cells)):
            lines[j][i] += "\n" * (max_cell_height - lines[j][i].count("\n"))
    cols = [_vbox(*col) for col in lines]

    return _hstack(cols)


def _layer_box(name: str, indim: Any = None, outdim: Any = None) -> str:
    """Create a keras-like layer diagram

    Args:
        name (str): layer name
        indim (Any): input shape. Defaults to None.
        outdim (Any): output shape. Defaults to None.

    Returns:
        str: box string

    Examples:
        >>> print(_layer_box("Test",(1,1,1),(1,1,1)))
        ┌──────┬────────┬───────────┐
        │      │ Input  │ (1, 1, 1) │
        │ Test │────────┼───────────┤
        │      │ Output │ (1, 1, 1) │
        └──────┴────────┴───────────┘

    """
    return _hstack(
        [
            _vbox(f"\n {name} \n"),
            _table([[" Input ", " Output "], [f" {indim} ", f" {outdim} "]]),
        ]
    )
