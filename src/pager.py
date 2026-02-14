from __future__ import annotations

from dataclasses import dataclass

ORDER = {"KOSPI": 0, "NASDAQ": 1, "CRYPTO": 2}


@dataclass
class GroupInfo:
    group: str
    score: int
    state: str
    benchmark: str


def rank_groups(groups: list[GroupInfo]) -> list[GroupInfo]:
    return sorted(groups, key=lambda g: (-g.score, ORDER[g.group]))


def compute_allocation(ranked: list[GroupInfo]) -> dict[str, int]:
    scores = {g.group: g.score for g in ranked}
    states = {g.group: g.state for g in ranked}
    groups = [g.group for g in ranked]

    if len(set(scores.values())) == 1:
        return {g: 5 for g in groups}

    bull = [g for g in groups if states[g] == "BULL"]
    turn = [g for g in groups if states[g] == "TURN"]
    bear = [g for g in groups if states[g] == "BEAR"]
    alloc = {g: 0 for g in groups}

    if len(bull) == 2:
        alloc[bull[0]] = 4
        alloc[bull[1]] = 4
        if turn and bear:
            alloc[turn[0]] = 4
            alloc[bear[0]] = 3
        elif turn:
            alloc[turn[0]] = 7
        elif bear:
            alloc[bear[0]] = 7
        return alloc

    if len(bull) == 1:
        alloc[bull[0]] = 8
        remainder = [g for g in groups if g != bull[0]]
        alloc[remainder[0]] = 4
        alloc[remainder[1]] = 3
        return alloc

    if len(turn) >= 2:
        alloc[turn[0]] = 4
        alloc[turn[1]] = 4
    elif len(turn) == 1:
        alloc[turn[0]] = 8

    remaining = 15 - sum(alloc.values())
    for g in groups:
        if remaining <= 0:
            break
        if states[g] == "BEAR":
            alloc[g] += remaining
            remaining = 0

    idx = 0
    while remaining > 0:
        alloc[groups[idx % len(groups)]] += 1
        remaining -= 1
        idx += 1
    return alloc


def _take(group: str, bucket: str, n: int, lists: dict, cursors: dict) -> list[dict]:
    if n <= 0:
        return []
    arr = lists[group][bucket]
    cur = cursors[group][bucket]
    selected = arr[cur : cur + n]
    cursors[group][bucket] = cur + len(selected)
    return selected


def build_page(ranked: list[GroupInfo], allocation: dict[str, int], lists: dict, cursors: dict) -> tuple[list[dict], dict]:
    items: list[dict] = []
    deficits: list[tuple[str, int]] = []

    for gi in ranked:
        group = gi.group
        slots = allocation.get(group, 0)
        bucket = "bull" if gi.state == "BULL" else "bear"
        selected = _take(group, bucket, slots, lists, cursors)
        items.extend(selected)
        if len(selected) < slots:
            deficits.append((bucket, slots - len(selected)))

    for bucket, need in deficits:
        if len(items) >= 15:
            break
        for gi in ranked:
            if need <= 0 or len(items) >= 15:
                break
            selected = _take(gi.group, bucket, need, lists, cursors)
            items.extend(selected)
            need -= len(selected)

    return items[:15], cursors
