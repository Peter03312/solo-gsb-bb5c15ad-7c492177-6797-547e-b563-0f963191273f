"""持久层：草稿（带乐观锁修订号、按键合并与修订快照）、不可变发布版本、默认投递策略（基线/候选灰度、按修订号的熔断设置与熔断记录）、幂等投递判定与历史判定核对报表。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .routing import Rule, RuleSet, Target


class RevisionConflict(Exception):
    """所见修订号与当前修订号不一致（草稿或默认投递策略）。"""

    def __init__(self, current: int, subject: str = "draft"):
        super().__init__(f"revision conflict: current {subject} revision is {current}")
        self.current = current


class RevisionSnapshotUnavailable(Exception):
    """无法取得客户端所见修订的快照（如旧库未留存该修订）：明确拒绝合并。"""

    def __init__(self, revision: int, current: int):
        super().__init__(f"revision snapshot unavailable: {revision} (current revision is {current})")
        self.revision = revision
        self.current = current


class MergeConflict(RevisionConflict):
    """落后修订合并时，本次操作涉及的键中有相对所见快照已改变的键。"""

    def __init__(self, current: int, conflict_keys: list[dict]):
        super().__init__(current)
        self.conflict_keys = conflict_keys


class DraftRuleNotFound(Exception):
    """删除操作引用了当前草稿中不存在的规则。"""

    def __init__(self, match_type: str, match_key: str):
        super().__init__(f"rule not found in draft: {match_type} {match_key}")
        self.match_type = match_type
        self.match_key = match_key


class PolicyVersionNotFound(Exception):
    """策略引用了不存在的已发布版本。"""

    def __init__(self, version: int):
        super().__init__(f"published version not found: {version}")
        self.version = version


class BreakerRevisionConflict(Exception):
    """熔断设置所针对的修订号不是当前策略修订号（须针对新修订重新设置）。"""

    def __init__(self, revision: int, current: int):
        super().__init__(
            f"circuit-breaker revision conflict: {revision} is not the current policy revision ({current})"
        )
        self.revision = revision
        self.current = current


class PolicyNotConfigured(Exception):
    """尚未配置默认投递策略时设置熔断：明确拒绝，不产生任何熔断设置。"""


def _row_to_rule(row: sqlite3.Row) -> Rule:
    targets = tuple(Target(kind=t["kind"], value=t["value"]) for t in json.loads(row["targets"]))
    return Rule(
        match_type=row["match_type"],
        match_key=row["match_key"],
        action=row["action"],
        targets=targets,
    )


def _rule_from_doc(doc: dict) -> Rule:
    return Rule(
        match_type=doc["match_type"],
        match_key=doc["match_key"],
        action=doc["action"],
        targets=tuple(Target(kind=t["kind"], value=t["value"]) for t in doc.get("targets", [])),
    )


def rule_to_dict(rule: Rule) -> dict:
    return {
        "match_type": rule.match_type,
        "match_key": rule.match_key,
        "action": rule.action,
        "targets": [{"kind": t.kind, "value": t.value} for t in rule.targets],
    }


def _save_draft_snapshot(conn: sqlite3.Connection, revision: int, rules: list[Rule]) -> None:
    """把某次草稿变更的完整规则快照写入 draft_revisions。须在事务内调用。"""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO draft_revisions (revision, rules, rule_count, created_at) VALUES (?, ?, ?, ?)",
        (revision, json.dumps([rule_to_dict(r) for r in rules]), len(rules), created_at),
    )


def get_draft(conn: sqlite3.Connection) -> tuple[int, list[Rule]]:
    revision = conn.execute("SELECT revision FROM draft_meta WHERE id = 1").fetchone()["revision"]
    rows = conn.execute(
        "SELECT match_type, match_key, action, targets FROM draft_rules ORDER BY match_type, match_key"
    ).fetchall()
    return revision, [_row_to_rule(r) for r in rows]


def replace_draft(conn: sqlite3.Connection, expected_revision: int, rules: list[Rule]) -> int:
    """整体替换草稿；修订号不匹配抛 RevisionConflict。须在事务内调用。

    成功后留存完整修订快照（与按键合并一样可核对）。
    """
    revision, _ = get_draft(conn)
    if revision != expected_revision:
        raise RevisionConflict(revision)
    conn.execute("DELETE FROM draft_rules")
    conn.executemany(
        "INSERT INTO draft_rules (match_type, match_key, action, targets) VALUES (?, ?, ?, ?)",
        [
            (
                r.match_type,
                r.match_key,
                r.action,
                json.dumps([{"kind": t.kind, "value": t.value} for t in r.targets]),
            )
            for r in rules
        ],
    )
    new_revision = revision + 1
    conn.execute("UPDATE draft_meta SET revision = ? WHERE id = 1", (new_revision,))
    _save_draft_snapshot(conn, new_revision, rules)
    return new_revision


def _rules_by_key(rules: list[Rule]) -> dict[tuple[str, str], Rule]:
    return {(r.match_type, r.match_key): r for r in rules}


def get_draft_at(conn: sqlite3.Connection, revision: int) -> tuple[int, list[Rule]] | None:
    """读取某个历史草稿修订的完整快照；不存在返回 None。"""
    row = conn.execute(
        "SELECT revision, rules FROM draft_revisions WHERE revision = ?", (revision,)
    ).fetchone()
    if row is None:
        return None
    return row["revision"], [_rule_from_doc(doc) for doc in json.loads(row["rules"])]


def list_draft_revisions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT revision, rule_count, created_at FROM draft_revisions ORDER BY revision"
    ).fetchall()
    return [dict(r) for r in rows]


def merge_draft(
    conn: sqlite3.Connection,
    expected_revision: int,
    changes: list[tuple[str, str, str, Rule | None]],
) -> tuple[int, list[Rule]]:
    """把一组按键设置 / 删除操作合并进当前草稿。须在写事务内调用。

    - 取得到所见修订（expected_revision）的快照后逐键核对：仅当本次涉及的每个键
      相对该快照在当前草稿中均未改变（存在性与完整内容一致）时才合并；
      否则抛 MergeConflict（附全部冲突键与当前修订号），草稿不变。
    - 所见修订即当前修订时，键必然“未改变”，直接合并。
    - 无法取得所见修订快照（RevisionSnapshotUnavailable）时明确拒绝合并。
    - 删除当前草稿中不存在的规则抛 DraftRuleNotFound（整组拒绝，草稿不变）。
    成功则写入合并结果、修订号 +1，并留存完整修订快照。
    """
    revision, current_rules = get_draft(conn)
    current_by_key = _rules_by_key(current_rules)

    if revision != expected_revision:
        seen = get_draft_at(conn, expected_revision)
        if seen is None:
            raise RevisionSnapshotUnavailable(expected_revision, revision)
        seen_by_key = _rules_by_key(seen[1])
        conflict_keys: list[dict] = []
        for _op, match_type, key, _rule in changes:
            ident = (match_type, key)
            seen_rule = seen_by_key.get(ident)
            current_rule = current_by_key.get(ident)
            if seen_rule != current_rule:  # 含“快照存在 / 当前不存在”与反向新增
                conflict_keys.append({"match_type": match_type, "match_key": key})
        if conflict_keys:
            raise MergeConflict(revision, conflict_keys)

    # 键相对所见快照均未改变（或所见即当前）：删除前先确认规则当前存在
    for _op, match_type, key, _rule in changes:
        if _op == "delete" and (match_type, key) not in current_by_key:
            raise DraftRuleNotFound(match_type, key)

    merged = dict(current_by_key)
    for op, match_type, key, rule in changes:
        ident = (match_type, key)
        if op == "delete":
            del merged[ident]
        else:
            assert rule is not None
            merged[ident] = rule
    new_rules = sorted(merged.values(), key=lambda r: (r.match_type, r.match_key))

    conn.execute("DELETE FROM draft_rules")
    conn.executemany(
        "INSERT INTO draft_rules (match_type, match_key, action, targets) VALUES (?, ?, ?, ?)",
        [
            (
                r.match_type,
                r.match_key,
                r.action,
                json.dumps([{"kind": t.kind, "value": t.value} for t in r.targets]),
            )
            for r in new_rules
        ],
    )
    new_revision = revision + 1
    conn.execute("UPDATE draft_meta SET revision = ? WHERE id = 1", (new_revision,))
    _save_draft_snapshot(conn, new_revision, new_rules)
    return new_revision, new_rules


def publish(conn: sqlite3.Connection, expected_revision: int, rules: list[Rule]) -> int:
    """把当前草稿固化为不可变版本，返回版本号。须在事务内调用。"""
    revision, _ = get_draft(conn)
    if revision != expected_revision:
        raise RevisionConflict(revision)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO versions (source_revision, created_at) VALUES (?, ?)",
        (revision, created_at),
    )
    version = cur.lastrowid
    conn.executemany(
        "INSERT INTO version_rules (version, match_type, match_key, action, targets) VALUES (?, ?, ?, ?, ?)",
        [
            (
                version,
                r.match_type,
                r.match_key,
                r.action,
                json.dumps([{"kind": t.kind, "value": t.value} for t in r.targets]),
            )
            for r in rules
        ],
    )
    return version


def list_versions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT v.version, v.source_revision, v.created_at, COUNT(r.match_key) AS rule_count
        FROM versions v LEFT JOIN version_rules r ON r.version = v.version
        GROUP BY v.version ORDER BY v.version
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_version(conn: sqlite3.Connection, version: int) -> dict | None:
    row = conn.execute(
        "SELECT version, source_revision, created_at FROM versions WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        return None
    rules = conn.execute(
        "SELECT match_type, match_key, action, targets FROM version_rules "
        "WHERE version = ? ORDER BY match_type, match_key",
        (version,),
    ).fetchall()
    return {**dict(row), "rules": [_row_to_rule(r) for r in rules]}


def load_ruleset(conn: sqlite3.Connection, version: int) -> RuleSet | None:
    found = get_version(conn, version)
    if found is None:
        return None
    return RuleSet(found["rules"])


# ---------- 默认投递策略（基线 / 候选灰度） ----------

def get_policy(conn: sqlite3.Connection) -> dict | None:
    """读取当前默认投递策略；从未配置时返回 None（修订号视为 0）。"""
    row = conn.execute(
        "SELECT revision, baseline_version, candidate_version, candidate_percent, updated_at "
        "FROM delivery_policy WHERE id = 1"
    ).fetchone()
    return dict(row) if row is not None else None


def replace_policy(
    conn: sqlite3.Connection,
    expected_revision: int,
    baseline_version: int,
    candidate_version: int,
    candidate_percent: int,
) -> dict:
    """在同一事务内核验所见策略修订号并保存新策略。

    - 未配置策略时修订号为 0，行尚不存在；
    - 修订号不匹配抛 RevisionConflict，调用方回滚，原策略保持不变；
    - 基线 / 候选必须都是已发布版本，否则抛 PolicyVersionNotFound（同样不改原策略）。
    """
    current = get_policy(conn)
    current_revision = current["revision"] if current is not None else 0
    if current_revision != expected_revision:
        raise RevisionConflict(current_revision, subject="policy")
    for version in (baseline_version, candidate_version):
        if conn.execute("SELECT 1 FROM versions WHERE version = ?", (version,)).fetchone() is None:
            raise PolicyVersionNotFound(version)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO delivery_policy (id, revision, baseline_version, candidate_version, candidate_percent, updated_at)
        VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            revision = excluded.revision,
            baseline_version = excluded.baseline_version,
            candidate_version = excluded.candidate_version,
            candidate_percent = excluded.candidate_percent,
            updated_at = excluded.updated_at
        """,
        (current_revision + 1, baseline_version, candidate_version, candidate_percent, updated_at),
    )
    return {
        "revision": current_revision + 1,
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "candidate_percent": candidate_percent,
        "updated_at": updated_at,
    }


# ---------- 灰度熔断（按策略修订号设置候选侧失败率阈值，达标即把候选比例置 0 并递增修订号） ----------

def get_breaker_setting(conn: sqlite3.Connection, policy_revision: int) -> dict | None:
    """读取某个策略修订号上的熔断设置；未设置返回 None。"""
    row = conn.execute(
        "SELECT policy_revision, failure_threshold, min_samples, window_minutes, created_at, updated_at "
        "FROM breaker_settings WHERE policy_revision = ?",
        (policy_revision,),
    ).fetchone()
    return dict(row) if row is not None else None


def put_breaker_setting(
    conn: sqlite3.Connection,
    expected_revision: int,
    failure_threshold: int,
    min_samples: int,
    window_minutes: int,
) -> dict:
    """针对当前策略修订号设置 / 更新熔断参数。须在写事务内调用。

    只能针对“当前”策略修订号设置：熔断触发或人工更新策略后修订号已前进，
    针对旧修订号的设置请求抛 BreakerRevisionConflict（不产生任何变更），
    操作方须取到新修订号后重新设置。同修订号重复设置为幂等更新（不产生熔断）。
    """
    current = get_policy(conn)
    if current is None:
        raise PolicyNotConfigured
    if current["revision"] != expected_revision:
        raise BreakerRevisionConflict(expected_revision, current["revision"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO breaker_settings (
            policy_revision, failure_threshold, min_samples, window_minutes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(policy_revision) DO UPDATE SET
            failure_threshold = excluded.failure_threshold,
            min_samples = excluded.min_samples,
            window_minutes = excluded.window_minutes,
            updated_at = excluded.updated_at
        """,
        (expected_revision, failure_threshold, min_samples, window_minutes, now, now),
    )
    return get_breaker_setting(conn, expected_revision)


def breaker_already_tripped(conn: sqlite3.Connection, policy_revision: int) -> bool:
    """该策略修订号是否已熔断过（每个修订号至多触发一次）。"""
    return conn.execute(
        "SELECT 1 FROM breaker_trips WHERE policy_revision = ?", (policy_revision,)
    ).fetchone() is not None


def breaker_candidate_window_stats(
    conn: sqlite3.Connection,
    *,
    policy_revision: int,
    window_start: str,
) -> tuple[int, int]:
    """统计同一策略修订号、观察窗口起点（含）之后已持久化的候选侧判定。

    只统计候选侧（selection='candidate'）判定；成功与失败（rejected/no_route/cycle）
    均计入样本，失败数仅含 status='failed'。重放不产生新行，故天然不重复计数。
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failures
        FROM delivery_decisions
        WHERE policy_revision = ? AND selection = 'candidate' AND created_at >= ?
        """,
        (policy_revision, window_start),
    ).fetchone()
    return row["total"], row["failures"] or 0


def trip_breaker(
    conn: sqlite3.Connection,
    *,
    old_revision: int,
    sample_size: int,
    failure_count: int,
    setting: dict,
    tripped_at: str,
) -> dict | None:
    """执行熔断：候选比例置 0、策略修订号 +1，并留档熔断记录。须在写事务内调用。

    以修订号为条件更新策略：BEGIN IMMEDIATE 串行化下策略不可能在本事务期间被他人推进
    （策略行在事务开头读出）；防御性地，若条件更新 0 行则放弃熔断（返回 None），
    已持久化的首次判定照常提交，绝不因熔断异常而回滚判定。
    breaker_trips 的 UNIQUE(policy_revision) 约束兜底保证同一修订号只熔断一次。
    """
    cur = conn.execute(
        "UPDATE delivery_policy SET revision = ?, candidate_percent = 0, updated_at = ? "
        "WHERE id = 1 AND revision = ?",
        (old_revision + 1, tripped_at, old_revision),
    )
    if cur.rowcount != 1:  # 防御：串行化下不可达；宁可不熔断也不回滚首次判定
        return None
    conn.execute(
        """
        INSERT INTO breaker_trips (
            policy_revision, new_revision, sample_size, failure_count,
            failure_threshold, min_samples, window_minutes, tripped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            old_revision, old_revision + 1, sample_size, failure_count,
            setting["failure_threshold"], setting["min_samples"], setting["window_minutes"], tripped_at,
        ),
    )
    return {
        "policy_revision": old_revision,
        "new_revision": old_revision + 1,
        "sample_size": sample_size,
        "failure_count": failure_count,
        "failure_threshold": setting["failure_threshold"],
        "min_samples": setting["min_samples"],
        "window_minutes": setting["window_minutes"],
        "tripped_at": tripped_at,
    }


def list_breaker_trips(conn: sqlite3.Connection, policy_revision: int | None = None) -> list[dict]:
    """列出熔断记录（默认全部，按触发顺序）；可按（旧）策略修订号筛选。"""
    if policy_revision is None:
        rows = conn.execute(
            "SELECT policy_revision, new_revision, sample_size, failure_count, "
            "failure_threshold, min_samples, window_minutes, tripped_at "
            "FROM breaker_trips ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT policy_revision, new_revision, sample_size, failure_count, "
            "failure_threshold, min_samples, window_minutes, tripped_at "
            "FROM breaker_trips WHERE policy_revision = ? ORDER BY id",
            (policy_revision,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- 幂等投递判定（投递编号 → 首次判定，永久可查、原样重放） ----------

def get_decision(conn: sqlite3.Connection, delivery_id: str) -> dict | None:
    """按投递编号读取已持久化的判定；不存在返回 None。JSON 字段在此还原。"""
    row = conn.execute(
        "SELECT delivery_id, address, policy_revision, baseline_version, candidate_version, "
        "candidate_percent, selection, version, status, endpoints, error, created_at "
        "FROM delivery_decisions WHERE delivery_id = ?",
        (delivery_id,),
    ).fetchone()
    if row is None:
        return None
    decision = dict(row)
    decision["endpoints"] = json.loads(decision["endpoints"]) if decision["endpoints"] is not None else None
    decision["error"] = json.loads(decision["error"]) if decision["error"] is not None else None
    return decision


def save_decision(
    conn: sqlite3.Connection,
    *,
    delivery_id: str,
    address: str,
    policy: dict,
    selection: str,
    version: int,
    status: str,
    endpoints: list[dict] | None,
    error: dict | None,
) -> dict:
    """把首次判定整体持久化（成功含全部终点路径，失败含原错误详情）。须在事务内调用。

    判定一旦写入不再改写：策略后续变更不影响已持久化的结果。
    """
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO delivery_decisions (
            delivery_id, address, policy_revision, baseline_version, candidate_version,
            candidate_percent, selection, version, status, endpoints, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            delivery_id,
            address,
            policy["revision"],
            policy["baseline_version"],
            policy["candidate_version"],
            policy["candidate_percent"],
            selection,
            version,
            status,
            json.dumps(endpoints) if endpoints is not None else None,
            json.dumps(error) if error is not None else None,
            created_at,
        ),
    )
    return get_decision(conn, delivery_id)


# ---------- 历史投递判定核对报表（时间窗 + 快照边界 + 分组计数 + 键集分页） ----------
#
# 报表在 delivery_decisions 的 rowid（随提交单调递增的内部序号）上固定边界：
# 首次查询记下当时的最大序号，分组、总数与全部翻页都只看序号不超过该边界的判定，
# 因此并发新投递不会进入旧报表；按 rowid 键集分页（唯一、单调）保证页间不漏不重。
# delivery_id 是主键，同一编号天然只计一次；已持久化的失败判定同样计入。

def _report_where(
    start: str,
    end: str,
    policy_revision: int | None,
    snapshot_seq: int,
    after_seq: int = 0,
) -> tuple[str, list]:
    """报表共用的筛选：同一快照边界 + 左闭右开时间窗（+ 可选策略修订号 + 分页下界）。"""
    clause = "rowid <= ? AND rowid > ? AND created_at >= ? AND created_at < ?"
    params: list = [snapshot_seq, after_seq, start, end]
    if policy_revision is not None:
        clause += " AND policy_revision = ?"
        params.append(policy_revision)
    return clause, params


def report_snapshot_seq(conn: sqlite3.Connection) -> int:
    """截至当前已提交判定的内部序号上界；报表以此固定边界，之后的新投递不进入该报表。"""
    row = conn.execute("SELECT COALESCE(MAX(rowid), 0) AS seq FROM delivery_decisions").fetchone()
    return row["seq"]


def report_groups(
    conn: sqlite3.Connection,
    *,
    start: str,
    end: str,
    policy_revision: int | None,
    snapshot_seq: int,
) -> list[dict]:
    """按选中版本、分流侧（基线/候选）、结果（ok 或失败原因）分组计数。"""
    clause, params = _report_where(start, end, policy_revision, snapshot_seq)
    rows = conn.execute(
        f"""
        SELECT version, selection,
               CASE WHEN status = 'ok' THEN 'ok' ELSE json_extract(error, '$.error') END AS outcome,
               COUNT(*) AS count
        FROM delivery_decisions
        WHERE {clause}
        GROUP BY version, selection, outcome
        ORDER BY version, selection, outcome
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def report_total(
    conn: sqlite3.Connection,
    *,
    start: str,
    end: str,
    policy_revision: int | None,
    snapshot_seq: int,
) -> int:
    """与分组同一筛选下的判定总数（恒等于各组计数之和）。"""
    clause, params = _report_where(start, end, policy_revision, snapshot_seq)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM delivery_decisions WHERE {clause}", params
    ).fetchone()
    return row["n"]


def report_items(
    conn: sqlite3.Connection,
    *,
    start: str,
    end: str,
    policy_revision: int | None,
    snapshot_seq: int,
    after_seq: int,
    limit: int,
) -> list[dict]:
    """取一页判定明细（按内部序号升序，从 after_seq 之后继续，最多 limit 条）。"""
    clause, params = _report_where(start, end, policy_revision, snapshot_seq, after_seq)
    rows = conn.execute(
        f"""
        SELECT rowid AS seq, delivery_id, address, policy_revision, version, selection, status,
               CASE WHEN status = 'ok' THEN NULL ELSE json_extract(error, '$.error') END AS error_type,
               created_at
        FROM delivery_decisions
        WHERE {clause}
        ORDER BY rowid
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [dict(r) for r in rows]
