"""灰度策略按候选侧投递失败率自动熔断（PUT/GET /policy/circuit-breaker、GET /policy/circuit-breaker/trips）：

- 操作方凭当前策略修订号设置失败率阈值（1–100%）、最少候选判定数、观察窗口（分钟，正整数）；
- 每次首次提交投递判定后，只统计同一策略修订号在窗口内已持久化的候选侧判定；
- 样本达下限且失败占比达到阈值时只触发一次：候选比例置 0、策略修订号 +1，
  记录旧修订号、样本数、失败数与触发时间；触发请求仍返回原判定；
- 同编号重放不重复计数，历史判定及响应不改写；后续默认投递按基线执行；
- 人工更新策略后旧设置失效，须针对新修订重新设置；
- 未配置策略、非法参数、修订号冲突明确报错且不产生熔断；
- 并发提交与人工更新不能覆盖彼此结果。
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

# 必须在导入 app 之前设置，config 在导入时读取
os.environ.setdefault("ROUTER_DB", os.path.join(tempfile.mkdtemp(prefix="router-test-"), "test.db"))

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture()
def client():
    db.init_db()
    conn = db.connect()
    conn.executescript(
        "DELETE FROM delivery_decisions;"
        "DELETE FROM breaker_trips;"
        "DELETE FROM breaker_settings;"
        "DELETE FROM delivery_policy;"
        "DELETE FROM version_rules;"
        "DELETE FROM versions;"
        "DELETE FROM draft_rules;"
        "DELETE FROM draft_revisions;"
        "DELETE FROM sqlite_sequence WHERE name = 'versions';"
        "DELETE FROM sqlite_sequence WHERE name = 'breaker_trips';"
        "INSERT INTO draft_revisions (revision, rules, rule_count, created_at)"
        " VALUES (0, '[]', 0, strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'));"
        "UPDATE draft_meta SET revision = 0 WHERE id = 1;"
    )
    conn.close()
    with TestClient(app) as c:
        yield c


BASE_RULES = [
    {
        "match_type": "exact",
        "match_key": "shared@corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "base@mail.example.com"}],
    },
    {
        "match_type": "domain",
        "match_key": "corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "base-catchall@mail.example.com"}],
    },
]

CANDIDATE_RULES = [
    {
        "match_type": "exact",
        "match_key": "shared@corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "cand@mail.example.com"}],
    },
    # 候选里 blocked@ 命中拒收；候选无域兜底 → 其他地址 no_route
    {
        "match_type": "exact",
        "match_key": "blocked@corp.example.com",
        "action": "reject",
        "targets": [],
    },
]


def put_rules(client, revision, rules):
    return client.put("/draft", json={"revision": revision, "rules": rules})


def publish(client, revision):
    return client.post("/publish", json={"revision": revision})


def publish_two_versions(client):
    assert put_rules(client, 0, BASE_RULES).status_code == 200
    assert publish(client, 1).status_code == 200
    assert put_rules(client, 1, CANDIDATE_RULES).status_code == 200
    resp = publish(client, 2)
    assert resp.status_code == 200, resp.text
    return 1, 2


def set_policy(client, revision, baseline, candidate, percent):
    return client.put("/policy", json={
        "revision": revision,
        "baseline_version": baseline,
        "candidate_version": candidate,
        "candidate_percent": percent,
    })


def submit(client, delivery_id, address):
    return client.post("/deliveries", json={"delivery_id": delivery_id, "address": address})


def set_breaker(client, revision, threshold=50, min_samples=2, window=60):
    return client.put("/policy/circuit-breaker", json={
        "revision": revision,
        "failure_threshold": threshold,
        "min_samples": min_samples,
        "window_minutes": window,
    })


def trips(client, revision=None):
    url = "/policy/circuit-breaker/trips"
    if revision is not None:
        url += f"?policy_revision={revision}"
    return client.get(url).json()["trips"]


def backdate_decisions(ids_minutes_ago: list[tuple[str, int]]) -> None:
    """直接在库里把已持久化判定的 created_at 前移若干分钟（观察窗口测试用）。"""
    conn = db.connect()
    try:
        for delivery_id, minutes in ids_minutes_ago:
            ts = (
                (datetime.now(timezone.utc) - timedelta(minutes=minutes))
                .replace(microsecond=0)
                .isoformat()
            )
            conn.execute(
                "UPDATE delivery_decisions SET created_at = ? WHERE delivery_id = ?",
                (ts, delivery_id),
            )
    finally:
        conn.close()


@pytest.fixture()
def gray(client):
    """已发布 v1 基线 / v2 候选，策略 revision 1 = 100% 候选。"""
    v1, v2 = publish_two_versions(client)
    assert set_policy(client, 0, v1, v2, 100).status_code == 200
    return v1, v2


# ---------- 设置与查询 ----------

def test_set_and_get_breaker_for_current_revision(client, gray):
    resp = set_breaker(client, 1, threshold=40, min_samples=5, window=30)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "policy_revision": 1,
        "failure_threshold": 40,
        "min_samples": 5,
        "window_minutes": 30,
        "created_at": body["created_at"],
        "updated_at": body["updated_at"],
    }
    assert body["created_at"] and body["updated_at"]
    got = client.get("/policy/circuit-breaker")
    assert got.status_code == 200
    assert got.json() == body
    got_rev = client.get("/policy/circuit-breaker?policy_revision=1")
    assert got_rev.status_code == 200 and got_rev.json() == body


def test_set_breaker_is_idempotent_update_on_same_revision(client, gray):
    assert set_breaker(client, 1, threshold=80, min_samples=10, window=15).status_code == 200
    resp = set_breaker(client, 1, threshold=25, min_samples=3, window=5)
    assert resp.status_code == 200
    body = resp.json()
    assert (body["failure_threshold"], body["min_samples"], body["window_minutes"]) == (25, 3, 5)
    # 重复设置不产生熔断记录、不推进策略修订号
    assert client.get("/policy").json()["revision"] == 1
    assert trips(client) == []


def test_get_breaker_without_settings_404(client, gray):
    resp = client.get("/policy/circuit-breaker")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["error"] == "breaker_not_configured"
    assert detail["policy_revision"] == 1
    assert detail["current_revision"] == 1
    # 查询从未有过设置的历史修订号同样 404
    resp2 = client.get("/policy/circuit-breaker?policy_revision=9")
    assert resp2.status_code == 404
    assert resp2.json()["detail"]["current_revision"] == 1


def test_set_breaker_without_policy_409_and_get_409(client):
    # 未发布 / 未配置策略
    resp = set_breaker(client, 1)
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "policy_not_configured"
    got = client.get("/policy/circuit-breaker")
    assert got.status_code == 409
    assert got.json()["detail"]["error"] == "policy_not_configured"
    # 明确报错，不产生熔断设置 / 记录
    assert trips(client) == []


def test_invalid_breaker_params_422_without_trip(client, gray):
    def put(payload):
        return client.put("/policy/circuit-breaker", json=payload)

    base = {"revision": 1, "failure_threshold": 50, "min_samples": 2, "window_minutes": 60}
    for patch in (
        {"failure_threshold": 0}, {"failure_threshold": 101}, {"failure_threshold": -5},
        {"min_samples": 0}, {"min_samples": -1},
        {"window_minutes": 0}, {"window_minutes": -30},
        {"revision": 0},
    ):
        payload = {**base, **patch}
        resp = put(payload)
        assert resp.status_code == 422, (patch, resp.text)
    # 缺字段 422
    assert put({"revision": 1, "min_samples": 2, "window_minutes": 60}).status_code == 422
    # 小数拒绝；无法解析为整数的字符串拒绝（与既有 /policy 一致，纯数字字符串由模型规范化）
    assert put({**base, "min_samples": 1.5}).status_code == 422
    assert put({**base, "window_minutes": "abc"}).status_code == 422
    # 全部非法参数都未写库，且没有熔断
    assert client.get("/policy/circuit-breaker").status_code == 404
    assert trips(client) == []


def test_set_breaker_stale_revision_conflict_409(client, gray):
    # 针对旧修订号（0 从未存在策略、2 尚未产生）设置 → 409 修订号冲突
    resp = set_breaker(client, 2)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "breaker_revision_conflict"
    assert detail["revision"] == 2 and detail["current_revision"] == 1
    # 冲突不写库
    assert client.get("/policy/circuit-breaker").status_code == 404


# ---------- 触发熔断 ----------

def test_breaker_trips_when_threshold_reached(client, gray):
    v1, v2 = gray
    set_breaker(client, 1, threshold=50, min_samples=2, window=60)
    # 第 1 个候选判定失败：样本不足，不熔断
    first = submit(client, "msg-1", "blocked@corp.example.com")
    assert first.status_code == 422  # 触发请求仍返回原判定（原失败状态）
    assert client.get("/policy").json()["revision"] == 1
    assert trips(client) == []
    # 第 2 个候选判定成功：样本 2、失败 1（50% 达标）→ 熔断一次
    second = submit(client, "msg-2", "shared@corp.example.com")
    assert second.status_code == 200  # 触发请求仍返回原判定（候选侧成功结果）
    policy = client.get("/policy").json()
    assert policy["revision"] == 2
    assert policy["candidate_percent"] == 0          # 候选比例置 0
    assert policy["baseline_version"] == v1 and policy["candidate_version"] == v2
    records = trips(client)
    assert len(records) == 1
    record = records[0]
    assert record == {
        "policy_revision": 1,                        # 旧修订号
        "new_revision": 2,
        "sample_size": 2,
        "failure_count": 1,
        "failure_threshold": 50,
        "min_samples": 2,
        "window_minutes": 60,
        "tripped_at": record["tripped_at"],
    }
    assert record["tripped_at"]
    # 按旧修订号筛选熔断记录
    assert trips(client, revision=1) == [record]
    assert trips(client, revision=2) == []


def test_triggering_request_returns_original_decision(client, gray):
    # 熔断在事务内、判定落库之后发生：响应仍是首次的候选侧判定
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    resp = submit(client, "msg-fail", "blocked@corp.example.com")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["selection"] == "candidate"
    assert detail["policy_revision"] == 1            # 响应里的修订号仍是受理时的旧修订
    assert detail["version"] == 2
    # 同编号重放逐字段返回相同的首次 422，不重复计数
    replay = submit(client, "msg-fail", "BLOCKED@CORP.EXAMPLE.COM")
    assert replay.status_code == 422
    assert replay.json() == resp.json()
    assert len(trips(client)) == 1


def test_after_trip_default_delivery_follows_baseline(client, gray):
    v1, v2 = gray
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    submit(client, "msg-1", "blocked@corp.example.com")  # 熔断 → revision 2, 0%
    assert client.get("/policy").json()["revision"] == 2
    # /deliver 即时分桶：候选比例 0 → 全部基线
    deliver = client.post("/deliver", json={"address": "shared@corp.example.com"})
    assert deliver.status_code == 200
    assert deliver.json()["selection"] == "baseline"
    assert deliver.json()["version"] == v1
    assert deliver.json()["policy_revision"] == 2
    # 新的首次投递判定也按新策略走基线
    ok = submit(client, "msg-2", "blocked@corp.example.com")
    assert ok.status_code == 200                      # 基线有域兜底，成功
    body = ok.json()
    assert body["selection"] == "baseline" and body["version"] == v1
    assert body["policy_revision"] == 2


def test_breaker_trips_only_once_per_revision(client, gray):
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    submit(client, "msg-1", "blocked@corp.example.com")   # 熔断
    assert client.get("/policy").json()["revision"] == 2
    # 修订号已前进：旧修订号 rev1 的窗口里再来候选判定也不会再次熔断
    # （新策略 0%，提交已全部走基线；直接核对 rev1 只有一条熔断记录）
    submit(client, "msg-2", "blocked@corp.example.com")
    assert client.get("/policy").json()["revision"] == 2  # 不继续递增
    assert len(trips(client)) == 1


def test_below_min_samples_does_not_trip(client, gray):
    set_breaker(client, 1, threshold=1, min_samples=5, window=60)
    for i in range(4):
        submit(client, f"msg-{i}", "blocked@corp.example.com")
    assert client.get("/policy").json()["revision"] == 1
    assert trips(client) == []
    # 第 5 个失败判定：样本 5、失败率 100% → 熔断
    submit(client, "msg-4", "blocked@corp.example.com")
    assert client.get("/policy").json()["revision"] == 2
    assert trips(client)[0]["sample_size"] == 5
    assert trips(client)[0]["failure_count"] == 5


def test_below_threshold_does_not_trip(client, gray):
    # 阈值 80%：2 失败 + 1 成功 = 66.7%，不熔断；再失败 → 75% 仍不熔断；第 4 失败 → 100% 熔断
    set_breaker(client, 1, threshold=80, min_samples=3, window=60)
    submit(client, "msg-f1", "blocked@corp.example.com")
    submit(client, "msg-f2", "blocked@corp.example.com")
    submit(client, "msg-ok", "shared@corp.example.com")
    assert trips(client) == []
    submit(client, "msg-f3", "blocked@corp.example.com")
    assert trips(client) == []                         # 3/4 = 75% < 80%
    submit(client, "msg-f4", "blocked@corp.example.com")
    records = trips(client)
    assert len(records) == 1
    assert records[0]["sample_size"] == 5 and records[0]["failure_count"] == 4


def test_threshold_boundary_inclusive(client, gray):
    # 阈值 50%：2 个判定中 1 个失败即 50% >= 50%，应熔断（整数比较，不四舍五入）
    set_breaker(client, 1, threshold=50, min_samples=2, window=60)
    submit(client, "msg-ok", "shared@corp.example.com")
    assert trips(client) == []
    submit(client, "msg-fail", "blocked@corp.example.com")
    assert len(trips(client)) == 1


def test_only_candidate_decisions_count(client, gray):
    # 策略 50% 时，基线侧判定不参与熔断统计
    v1, v2 = gray
    assert set_policy(client, 1, v1, v2, 50).status_code == 200
    set_breaker(client, 2, threshold=1, min_samples=1, window=60)
    # 找一批稳定落基线桶的地址（基线侧全部成功，本就不该触发）
    from app.traffic import select_version
    baseline_addrs = []
    i = 0
    while len(baseline_addrs) < 5:
        addr = f"base{i}@corp.example.com"
        if select_version(addr, 50) == "baseline":
            baseline_addrs.append(addr)
        i += 1
    for j, addr in enumerate(baseline_addrs):
        resp = submit(client, f"msg-b{j}", addr)
        assert resp.status_code == 200 and resp.json()["selection"] == "baseline"
    assert client.get("/policy").json()["revision"] == 2
    assert trips(client) == []


def test_window_excludes_old_decisions(client, gray):
    # min_samples=2、窗口 60 分钟：窗口外的失败判定不计入
    set_breaker(client, 1, threshold=1, min_samples=2, window=60)
    submit(client, "msg-old", "blocked@corp.example.com")
    backdate_decisions([("msg-old", 90)])              # 移到窗口外
    # 此刻窗口内只有 0 条候选判定，新失败仍只有 1 条 → 样本不足
    submit(client, "msg-new1", "blocked@corp.example.com")
    assert trips(client) == []
    # 再来一条窗口内失败：窗口内共 2 条 → 熔断，记录样本不含窗口外判定
    submit(client, "msg-new2", "blocked@corp.example.com")
    records = trips(client)
    assert len(records) == 1
    assert records[0]["sample_size"] == 2 and records[0]["failure_count"] == 2


def test_window_rolling_recovers_when_samples_expire(client, gray):
    # 窗口 60 分钟、min_samples=2、阈值 50%：1 旧失败滑出窗口后，1 新失败不达标
    set_breaker(client, 1, threshold=50, min_samples=2, window=60)
    submit(client, "msg-old", "blocked@corp.example.com")
    backdate_decisions([("msg-old", 90)])
    submit(client, "msg-new", "blocked@corp.example.com")
    assert trips(client) == []                         # 窗口内仅 1 条


def test_replay_does_not_recount(client, gray):
    # 同编号重放不产生新行：min_samples=3 时两次重放不会凑够样本
    set_breaker(client, 1, threshold=1, min_samples=3, window=60)
    submit(client, "msg-1", "blocked@corp.example.com")
    submit(client, "msg-1", "blocked@corp.example.com")
    submit(client, "msg-1", "blocked@corp.example.com")
    submit(client, "msg-2", "blocked@corp.example.com")
    submit(client, "msg-2", "blocked@corp.example.com")
    assert trips(client) == []
    conn = db.connect()
    n = conn.execute("SELECT COUNT(*) AS n FROM delivery_decisions").fetchone()["n"]
    conn.close()
    assert n == 2


def test_historical_decisions_and_responses_not_rewritten(client, gray):
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    first = submit(client, "msg-1", "blocked@corp.example.com")  # 熔断
    assert first.status_code == 422
    stored = client.get("/deliveries/msg-1").json()
    assert stored["status"] == "failed"
    assert stored["policy_revision"] == 1             # 历史判定保留旧修订号
    assert stored["candidate_percent"] == 100
    assert stored["selection"] == "candidate"
    # 历史报表仍可按旧修订号核对到这条失败
    report = client.get(
        "/deliveries/report"
        "?start=2000-01-01T00:00:00%2B00:00&end=2100-01-01T00:00:00%2B00:00&policy_revision=1"
    )
    assert report.status_code == 200
    assert report.json()["total"] == 1


# ---------- 人工更新策略后旧设置失效 ----------

def test_manual_policy_update_invalidates_old_breaker_settings(client, gray):
    v1, v2 = gray
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    # 人工把比例改为 50%（修订号推进到 2，未熔断）
    assert set_policy(client, 1, v1, v2, 50).status_code == 200
    # rev1 的旧设置不再服务当前修订
    current_setting = client.get("/policy/circuit-breaker")
    assert current_setting.status_code == 404
    detail = current_setting.json()["detail"]
    assert detail["error"] == "breaker_not_configured"
    assert detail["policy_revision"] == 2 and detail["current_revision"] == 2
    # 旧设置仍可按旧修订号查到留档
    old = client.get("/policy/circuit-breaker?policy_revision=1")
    assert old.status_code == 200 and old.json()["failure_threshold"] == 1
    # 候选失败不再触发熔断（新修订未设置）
    # 100% 分桶的 blocked@ 在 50% 下未必选候选；直接把比例调到 100 再失败
    assert set_policy(client, 2, v1, v2, 100).status_code == 200  # rev3
    resp = submit(client, "msg-1", "blocked@corp.example.com")
    assert resp.status_code == 422
    assert client.get("/policy").json()["revision"] == 3          # 修订号未因熔断前进
    assert trips(client) == []
    # 针对旧修订号设置 → 冲突；须针对新修订重新设置
    stale = set_breaker(client, 2)
    assert stale.status_code == 409
    assert stale.json()["detail"]["error"] == "breaker_revision_conflict"
    fresh = set_breaker(client, 3)
    assert fresh.status_code == 200
    submit(client, "msg-2", "blocked@corp.example.com")
    assert client.get("/policy").json()["revision"] == 4          # 重新设置后恢复熔断
    assert len(trips(client)) == 1
    assert trips(client)[0]["policy_revision"] == 3


def test_setting_revision_conflict_after_automatic_trip(client, gray):
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    submit(client, "msg-1", "blocked@corp.example.com")  # 熔断到 rev2
    # 熔断后须针对新修订重新设置：拿旧修订号 1 设置被拒
    stale = set_breaker(client, 1, threshold=10, min_samples=10, window=10)
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["error"] == "breaker_revision_conflict"
    assert detail["revision"] == 1 and detail["current_revision"] == 2
    # 针对新修订 2 设置成功
    assert set_breaker(client, 2, threshold=10, min_samples=10, window=10).status_code == 200


def test_breaker_does_not_fire_when_candidate_percent_zero(client, gray):
    v1, v2 = gray
    assert set_policy(client, 1, v1, v2, 0).status_code == 200  # rev2 = 0%
    set_breaker(client, 2, threshold=1, min_samples=1, window=60)
    for i in range(5):  # 全部基线，无候选判定，永不熔断
        resp = submit(client, f"msg-{i}", "blocked@corp.example.com")
        assert resp.status_code == 200 and resp.json()["selection"] == "baseline"
    assert client.get("/policy").json()["revision"] == 2
    assert trips(client) == []


# ---------- 并发：提交与人工更新不能覆盖彼此结果 ----------

def test_concurrent_submit_and_manual_update_do_not_overwrite(client, gray):
    v1, v2 = gray
    # min_samples=1：任意一个候选失败都会熔断（rev1 -> rev2, percent=0）
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)

    def manual_update():
        return set_policy(client, 1, v1, v2, 30)  # 所见 rev1 的人工更新

    def failing_submit():
        return submit(client, "msg-race", "blocked@corp.example.com")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda f: f(), [manual_update, failing_submit]))

    policy = client.get("/policy").json()
    # 两种串行顺序之一，但绝不会互相覆盖：
    #  A) 人工更新先（rev2=30%）：提交在 rev2 上首次受理（30% 分桶），rev2 无熔断设置
    #  B) 熔断先（rev2=0%）：人工更新 rev1 冲突 409，最终 rev2 percent=0
    final_percent = policy["candidate_percent"]
    assert final_percent in (0, 30)
    assert policy["revision"] == 2
    manual, submitted = results
    if final_percent == 0:
        # 熔断胜出：人工更新被 409 修订号冲突拒绝（而非被静默覆盖），提交按 rev1 候选失败
        assert manual.status_code == 409
        assert manual.json()["detail"]["error"] == "revision_conflict"
        assert submitted.status_code == 422
        stored = client.get("/deliveries/msg-race").json()
        assert stored["policy_revision"] == 1 and stored["selection"] == "candidate"
        assert len(trips(client)) == 1
    else:
        # 人工更新胜出：无熔断（rev2 未设置）；提交按 rev2 受理，30% 分桶决定基线 / 候选
        assert manual.status_code == 200
        assert submitted.status_code in (200, 422)
        assert trips(client) == []
        stored = client.get("/deliveries/msg-race").json()
        assert stored["policy_revision"] == 2


def test_concurrent_failing_submits_trip_exactly_once(client, gray):
    # 多个不同编号的失败提交并发：熔断只发生一次，修订号恰好 +1。
    # 首个候选失败判定落库即熔断（rev2 percent=0）；取得写锁更晚的请求按新策略走基线，
    # 返回自己的原判定（200）——触发与否都不改写各自的首次判定。
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(
            lambda i: submit(client, f"msg-c{i}", "blocked@corp.example.com"), range(8)
        ))
    assert all(r.status_code in (200, 422) for r in responses)
    assert any(r.status_code == 422 for r in responses)      # 至少一个候选侧失败触发了熔断
    policy = client.get("/policy").json()
    assert policy["revision"] == 2 and policy["candidate_percent"] == 0
    assert len(trips(client)) == 1
    record = trips(client)[0]
    # 记录的样本数等于 rev1 上持久化的候选判定数（1 至熔断提交的样本量），全部失败
    conn = db.connect()
    candidate_at_rev1 = conn.execute(
        "SELECT COUNT(*) AS n FROM delivery_decisions "
        "WHERE policy_revision = 1 AND selection = 'candidate'"
    ).fetchone()["n"]
    conn.close()
    assert record["sample_size"] == candidate_at_rev1
    assert record["failure_count"] == record["sample_size"]


# ---------- 熔断记录查询 ----------

def test_trips_empty_initially_and_filter_unknown_revision(client, gray):
    assert trips(client) == []
    assert client.get("/policy/circuit-breaker/trips?policy_revision=1").json()["trips"] == []


def test_trips_records_append_across_revisions(client, gray):
    v1, v2 = gray
    set_breaker(client, 1, threshold=1, min_samples=1, window=60)
    submit(client, "msg-1", "blocked@corp.example.com")     # rev1 -> rev2
    # 人工配置新比例后针对 rev2 重新设置并再次熔断
    assert set_policy(client, 2, v1, v2, 100).status_code == 200  # rev3
    set_breaker(client, 3, threshold=1, min_samples=1, window=60)
    submit(client, "msg-2", "blocked@corp.example.com")     # rev3 -> rev4
    records = trips(client)
    assert [r["policy_revision"] for r in records] == [1, 3]
    assert [r["new_revision"] for r in records] == [2, 4]
    assert client.get("/policy").json()["revision"] == 4
