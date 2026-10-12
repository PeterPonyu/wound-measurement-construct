"""Information sets: what each task permits and what each method variant needs.

T-MT (global missing time): the held time is removed for everyone; a method may
use the target individual's source and other observed times (the anchor) and all
other individuals' non-target times.  T-LIO (leave one individual out): the held
individual contributes only its source cells; other individuals contribute every
time, including the target time.
"""
from __future__ import annotations

import enum


class Info(enum.Flag):
    OWN_SOURCE = enum.auto()
    OWN_OTHER_TIMES = enum.auto()
    OTHERS_NONTARGET_TIMES = enum.auto()
    OTHERS_TARGET_TIME = enum.auto()


TASK_ALLOWED = {
    "T-MT": Info.OWN_SOURCE | Info.OWN_OTHER_TIMES | Info.OTHERS_NONTARGET_TIMES,
    "T-LIO": Info.OWN_SOURCE | Info.OTHERS_NONTARGET_TIMES | Info.OTHERS_TARGET_TIME,
}


class InformationSetViolation(ValueError):
    """A method needs information that the task does not permit."""


def allowed(kind: str) -> Info:
    try:
        return TASK_ALLOWED[kind]
    except KeyError as error:
        raise ValueError(f"unknown task kind {kind!r}") from error


def resolve_variant(kind: str, variants) -> str:
    permitted = allowed(kind)
    for name, needed in variants:
        if (needed & permitted) == needed:
            return name
    raise InformationSetViolation(f"no variant of {[name for name, _ in variants]} fits task {kind}")


def compatible(kind: str, variants) -> bool:
    try:
        resolve_variant(kind, variants)
    except InformationSetViolation:
        return False
    return True
