#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this file. If not, see <http://www.gnu.org/licenses/>.
#
#   Copyright © 2014-2023 The University of Tromsø & the Norwegian Sámi Parliament
#   http://giellatekno.uit.no & http://divvun.no
#
#   ADDENDUM: Pierre Beauguitte, National Library of Norway, 2023
#   Keep only methods used for classification
#
"""Utility functions and classes used by other modules in CorpusTools."""


from typing import Tuple
import operator
import os
import sys


def basename_noext(fname: str, ext: str) -> str:
    """Get the basename without the extension.

    Args:
        fname (str): path to the file.
        ext (str): the extension that should be removed.

    Returns:
        (str): fname without the extension.
    """
    return os.path.basename(fname)[: -len(ext)]


def sort_by_value(table: dict[str, float], reverse: bool = False) -> list[Tuple[str, float]]:
    """Sort the table by value.

    Args:
        table (dict): the dictionary that should be sorted.
        reverse (bool): whether or not to sort in reverse

    Returns:
        (dict): sorted by value.
    """
    return sorted(table.items(), key=operator.itemgetter(1), reverse=reverse)


def note(msg: str) -> None:
    """Print msg to stderr.

    Args:
        msg (str): the message
    """
    print(msg, file=sys.stderr)
