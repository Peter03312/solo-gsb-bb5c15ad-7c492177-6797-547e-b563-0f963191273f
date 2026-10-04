"""灰度熔断：候选侧投递失败率在观察窗口内达到阈值时，自动把候选比例置 0 并递增策略修订号。

触发判定（在 POST /deliveries 首次提交已落库的同一个写事务内执行）：

- 只在**首次提交**持久化一份候选侧判定之后评估；基线侧判定、相同编号重放不触发。
- 样本只取**同一策略修订号**、观察窗口起点（含）之后**已持久化**的候选侧判定，
  成功与失败（rejected/no_route/cycle）都计入样本，重放不产生新行故不重复计数。
- 样本数达到下限且失败数占比达到阈值（``failures * 100 >= threshold * total``，
  全程整数比较、不四舍五入）时熔断：每个修订号**只触发一次**，把候选比例置 0、
  策略修订号 +1，并记录旧修订号、样本数、失败数与触发时间。
- 触发熔断的那次请求仍返回**原判定**（含原错误状态），历史判定及响应不改写；
  后续默认投递按新策略（候选比例 0）执行，即全部基线。
- 熔断或人工更新策略产生新修订号后，旧修订号上的设置自然失效，须针对新修订重新设置。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from . import store


def evaluate_after_submit(
    conn: sqlite3.Connection,
    *,
    policy: dict,
    selection: str,
    now: datetime | None = None,
) -> dict | None:
    """首次提交持久化后评估是否需要熔断。须在写事务内、判定 INSERT 之后调用。

    返回熔断记录（dict）表示本次请求触发了熔断；返回 None 表示未触发。
    仅评估候选侧判定；``now`` 可注入（测试用），默认取当前 UTC 时刻，
    且与本次判定的 created_at 取自同一时刻，保证窗口边界与落库时间一致。
    """
    if selection != "candidate":
        return None
    revision = policy["revision"]
    setting = store.get_breaker_setting(conn, revision)
    if setting is None:
        return None
    if store.breaker_already_tripped(conn, revision):  # 每个修订号只触发一次
        return None

    moment = now or datetime.now(timezone.utc)
    window_start = (
        (moment - timedelta(minutes=setting["window_minutes"]))
        .replace(microsecond=0)
        .isoformat()
    )
    total, failures = store.breaker_candidate_window_stats(
        conn, policy_revision=revision, window_start=window_start
    )
    if total < setting["min_samples"]:
        return None
    if failures * 100 < setting["failure_threshold"] * total:
        return None

    tripped_at = moment.replace(microsecond=0).isoformat()
    return store.trip_breaker(
        conn,
        old_revision=revision,
        sample_size=total,
        failure_count=failures,
        setting=setting,
        tripped_at=tripped_at,
    )
