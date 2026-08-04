"""리포트 렌더링 — metrics dict를 사람이 읽는 마크다운으로 바꾼다.

print를 단계에서 걷어낸 대가로 이 모듈이 생긴다. 얻는 것:
  - 같은 지표로 마크다운·HTML·Slack 메시지를 각각 만들 수 있다
  - 리포트 형식을 바꿔도 분석 로직을 건드리지 않는다
  - 과거 실행의 metrics.json만 있으면 리포트를 다시 그릴 수 있다
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .quality import GateResult
from .storage import RunManifest

_DOW = ["월", "화", "수", "목", "금", "토", "일"]


def _fmt_actual(v: Any) -> str:
    """게이트 실측값 표시. 지표가 없을 수 있어(부분 실행) None을 견뎌야 한다."""
    if v is None:
        return "없음"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_report(cfg: Config, manifest: RunManifest, metrics: dict[str, dict],
                  notes: dict[str, list[str]], gates: list[GateResult],
                  rows_before: int) -> str:
    """실행 한 건의 리포트를 마크다운으로 만든다."""
    L: list[str] = []
    add = L.append

    add(f"# {cfg.name} 실행 리포트")
    add("")
    add(_table(["항목", "값"], [
        ["run_id", f"`{manifest.run_id}`"],
        ["실행 시각", manifest.started_at],
        ["입력", f"`{manifest.input_file}` (sha `{manifest.input_digest}`)"],
        ["설정", f"`{cfg.source_file.name}` (sha `{cfg.digest}`)"],
        ["상태", "**성공**" if manifest.status == "success" else "**실패**"],
        ["소요", f"{manifest.duration_sec}초" if manifest.duration_sec else "-"],
    ]))
    add("")

    # ---- 품질 게이트를 맨 앞에 둔다 ------------------------------------------
    # 자동 실행 결과에서 사람이 가장 먼저 봐야 할 것은 분석 내용이 아니라
    # "이번 실행을 믿어도 되는가"이다.
    add("## 품질 게이트")
    add("")
    if gates:
        add(_table(["항목", "종류", "기대", "실측", "결과"], [
            [g.name, g.kind,
             f"{g.expected:,}" if g.kind == "exact" else f"{g.expected[0]} ~ {g.expected[1]}",
             _fmt_actual(g.actual),
             "PASS" if g.passed else "**FAIL**"]
            for g in gates
        ]))
        n_fail = sum(1 for g in gates if not g.passed)
        add("")
        add(f"{len(gates)}개 중 {len(gates) - n_fail}개 통과"
            + (f", **{n_fail}개 실패**" if n_fail else " — 전부 통과"))
    else:
        add("설정된 게이트 없음")
    add("")

    # ---- 단계별 실행 요약 ----------------------------------------------------
    add("## 단계별 처리")
    add("")
    add(_table(["단계", "설명", "행 변화", "증감", "소요"], [
        [s["name"], s["description"],
         f"{s['rows_in']:,} → {s['rows_out']:,}",
         f"{s['rows_delta']:+,}" if s["rows_delta"] else "-",
         f"{s['duration_sec']}초"]
        for s in manifest.steps
    ]))
    final = manifest.steps[-1]["rows_out"] if manifest.steps else rows_before
    add("")
    add(f"원본 {rows_before:,}행 → 최종 {final:,}행 "
        f"(**{final / rows_before:.2%} 보존**)" if rows_before else "")
    add("")

    # ---- 각 단계가 남긴 근거 문장 --------------------------------------------
    add("## 처리 근거")
    add("")
    for step_name, lines in notes.items():
        if not lines:
            continue
        add(f"### {step_name}")
        add("")
        for line in lines:
            add(f"- {line}")
        add("")

    # ---- EDA 요약 ------------------------------------------------------------
    if prof := metrics.get("profile"):
        add("## 기본 EDA")
        add("")
        if desc := prof.get("describe"):
            add("### 수치형 기술통계")
            add("")
            add(_table(["컬럼", "평균", "중앙값", "95%", "99%", "최대"], [
                [col, f"{d['mean']:,.2f}", f"{d['50%']:,.2f}",
                 f"{d['95%']:,.2f}", f"{d['99%']:,.2f}", f"{d['max']:,.2f}"]
                for col, d in desc.items()
            ]))
            add("")

        if src := prof.get("by_record_source"):
            add("### record_source별 프로파일")
            add("")
            add(_table(["소스", "행 수", "평균 거리", "평균 요금", "평균 팁", "평균 총액"], [
                [k, f"{int(v['rows']):,}", f"{v['mean_distance']:.2f}",
                 f"{v['mean_fare']:.2f}", f"{v['mean_tip']:.2f}", f"{v['mean_total']:.2f}"]
                for k, v in src.items()
            ]))
            add("")

        if hours := prof.get("by_hour"):
            peak, quiet = prof.get("peak_hour"), prof.get("quietest_hour")
            add(f"### 시간대 (최다 {peak}시 / 최소 {quiet}시)")
            add("")
            mx = max(v["rows"] for v in hours.values())
            for h in sorted(hours, key=int):
                v = hours[h]
                bar = "█" * int(v["rows"] / mx * 30)
                add(f"- `{int(h):02d}시` {int(v['rows']):>8,}건 "
                    f"평균요금 {v['mean_fare']:>6.2f}  {bar}")
            add("")

        if dows := prof.get("by_dayofweek"):
            add("### 요일")
            add("")
            add(_table(["요일", "행 수", "평균 요금", "평균 팁"], [
                [_DOW[int(k)], f"{int(v['rows']):,}",
                 f"{v['mean_fare']:.2f}", f"{v['mean_tip']:.2f}"]
                for k, v in sorted(dows.items(), key=lambda x: int(x[0]))
            ]))
            add("")

        if buckets := prof.get("by_distance_bucket"):
            add("### 거리 구간별 요금 (마일당은 중앙값)")
            add("")
            add(_table(["구간", "행 수", "평균 요금", "마일당 요금"], [
                [k, f"{int(v['rows']):,}", f"{v['mean_fare']:.2f}",
                 f"{v['median_per_mile']:.2f}"]
                for k, v in buckets.items()
            ]))
            add("")

    add("---")
    add("")
    add(f"기준 문서: `결측치_중복_처리기준.md` · 설정: `{cfg.source_file.name}`")
    return "\n".join(L)


def render_console_summary(result_metrics: dict[str, dict],
                           gates: list[GateResult]) -> str:
    """터미널에 한 화면으로 띄우는 짧은 요약 (CLI 기본 출력)."""
    lines = []
    n_fail = sum(1 for g in gates if not g.passed)
    lines.append("품질 게이트: "
                 + (f"{len(gates)}개 전부 통과" if not n_fail
                    else f"{n_fail}개 실패 / {len(gates)}개"))
    for g in gates:
        if not g.passed:
            lines.append(f"  {g.describe()}")
    # 보존율은 '원본 대비'로 말한다. filter_outliers의 retention_ratio는 그 단계의
    # 입력 대비 값이라(중복 제거 후 기준) 여기 쓰면 실제보다 높게 보인다.
    fo = result_metrics.get("filter_outliers")
    am = result_metrics.get("analyze_missing")
    if fo:
        final = fo["rows_out"]
        origin = am.get("rows") if am else None
        tail = f" (원본 대비 보존율 {final / origin:.2%})" if origin else ""
        lines.append(f"최종 {final:,}행{tail}")
    return "\n".join(lines)
