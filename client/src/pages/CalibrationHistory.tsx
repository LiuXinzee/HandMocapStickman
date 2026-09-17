import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { Link } from "wouter";
import {
  HISTORY_EVENT,
  HISTORY_KEY,
  readHistory,
  importHistoryRounds,
  parseHistoryImport,
  historicalReading,
  historicalPoseDifference,
  downloadHistory,
  type CalibrationRound,
  type HistoricalSample,
} from "@/lib/calibrationHistory";
import {
  manualTargetLabel,
  MANUAL_AXES,
  MANUAL_MOTIONS,
  posePairKey,
} from "@/lib/manualOrientation";
import type { HandKey } from "@/lib/bendRange";

const button = "cyber-btn px-3 py-2 rounded-sm text-sm disabled:opacity-40";
const date = (at: string | null) =>
  at ? new Date(at).toLocaleString("zh-CN", { hour12: false }) : "未记录时间";
const degrees = (n: number) => n.toFixed(1) + "°";
const signed = (n: number) => (n > 0 ? "+" : "") + degrees(n);
const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function CalibrationHistory() {
  const [rounds, setRounds] = useState<CalibrationRound[]>([]);
  const [hand, setHand] = useState<HandKey>(
    new URLSearchParams(window.location.search).get("hand") === "LH"
      ? "LH"
      : "RH"
  );
  const [selectedIds, setSelectedIds] = useState<string[] | null>(null);
  const [axis, setAxis] = useState("all");
  const [target, setTarget] = useState("all");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const reload = () => {
    try {
      setRounds(readHistory());
      setError("");
    } catch (e) {
      setError("历史记录读取失败：" + errorText(e));
    }
  };
  useEffect(() => {
    reload();
    const storage = (e: StorageEvent) => {
      if (!e.key || e.key === HISTORY_KEY) reload();
    };
    window.addEventListener("storage", storage);
    window.addEventListener(HISTORY_EVENT, reload);
    window.addEventListener("focus", reload);
    return () => {
      window.removeEventListener("storage", storage);
      window.removeEventListener(HISTORY_EVENT, reload);
      window.removeEventListener("focus", reload);
    };
  }, []);
  const forHand = useMemo(
    () =>
      rounds
        .filter(r => r.hand === hand)
        .sort((a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt)),
    [rounds, hand]
  );
  const ids = selectedIds ?? forHand.slice(0, 4).map(r => r.id);
  const selected = forHand
    .filter(r => ids.includes(r.id))
    .sort((a, b) => Date.parse(a.startedAt) - Date.parse(b.startedAt));
  const targets = useMemo(() => {
    const unique = new Map<string, HistoricalSample>();
    for (const r of selected)
      for (const s of r.samples)
        if (axis === "all" || s.axis === axis) unique.set(posePairKey(s), s);
    return Array.from(unique.values()).sort(
      (a, b) => a.axis.localeCompare(b.axis) || a.degrees - b.degrees
    );
  }, [selected, axis]);
  const effectiveTarget = targets.some(s => posePairKey(s) === target)
    ? target
    : "all";
  const visible = targets.filter(
    s => effectiveTarget === "all" || posePairKey(s) === effectiveTarget
  );
  const details = selected
    .flatMap(round =>
      round.samples
        .filter(s => posePairKey(s) === effectiveTarget)
        .map(sample => ({ round, sample, ...historicalReading(round, sample) }))
    )
    .sort(
      (a, b) =>
        Date.parse(a.sample.recordedAt ?? a.round.startedAt) -
        Date.parse(b.sample.recordedAt ?? b.round.startedAt)
    );
  const importFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const imported = importHistoryRounds(
        parseHistoryImport(await file.text())
      );
      setHand(imported[0]?.hand ?? hand);
      setSelectedIds(
        imported
          .filter(r => r.hand === imported[0]?.hand)
          .slice(0, 6)
          .map(r => r.id)
      );
      setTarget("all");
      setMessage(
        "已导入 " +
          imported.length +
          " 轮记录到历史；这不会应用到手模。要试用，请回到手动角度校准，点击“载入最近一轮右手/左手历史”。相同文件重复导入不会重复，已有轮次的不同版本会另存保留。"
      );
      reload();
    } catch (e) {
      setError("导入失败：" + errorText(e));
    }
  };
  return (
    <main className="min-h-screen bg-[var(--hud-page)] text-[var(--hud-text)] p-4 sm:p-6 space-y-5 max-w-[1800px] mx-auto">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">校准历史 / 同姿势对比</h1>
          <p className="text-sm text-[var(--hud-soft)] mt-2">
            同一只手、同一动作、同一目标角度，比较每次手套报告的读数。
          </p>
        </div>
        <Link href="/mocap" className={button}>
          返回手套准备页
        </Link>
      </header>
      <div className="flex flex-wrap gap-2 items-center">
        <label>
          手别{" "}
          <select
            aria-label="历史手别"
            className="border p-2 bg-[var(--hud-page)]"
            value={hand}
            onChange={e => {
              setHand(e.target.value as HandKey);
              setSelectedIds(null);
              setTarget("all");
            }}
          >
            <option value="RH">右手</option>
            <option value="LH">左手</option>
          </select>
        </label>
        <button className={button} onClick={reload}>
          刷新历史
        </button>
        <label className={button}>
          导入校准 JSON
          <input
            aria-label="导入校准历史 JSON"
            type="file"
            accept=".json,application/json"
            className="sr-only"
            onChange={importFile}
          />
        </label>
        <button
          className={button}
          disabled={!rounds.length}
          onClick={() => downloadHistory(rounds)}
        >
          导出全部历史 JSON
        </button>
      </div>
      {error && (
        <p role="alert" className="text-[var(--hud-err)]">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      <p className="text-sm text-[var(--hud-soft)]">
        每次设零位开启一轮，每次记录都自动保存；重采和从校准中删除的旧读数也会保留。记录保存在当前浏览器、当前地址下。
        旧版未导出的采样无法追溯；已有“手动校准 JSON”可以导入。
      </p>
      {!forHand.length ? (
        <section className="border p-6 space-y-2">
          <h2 className="font-medium">这只手还没有历史记录</h2>
          <p>
            去手动角度校准中设零位并记录姿势，或导入之前导出的手动校准 JSON。
          </p>
        </section>
      ) : (
        <>
          <section className="border border-[var(--hud-line)] p-4 space-y-3">
            <h2 className="font-medium">选择要比较的轮次（最多同时 6 轮）</h2>
            <p className="text-sm text-[var(--hud-soft)]">
              默认最近 4
              轮；不同轮次都应按相同竖立姿势设零位，佩戴位置变化也会影响比较。
            </p>
            <div className="max-h-52 overflow-y-auto grid md:grid-cols-2 xl:grid-cols-3 gap-2">
              {forHand.map(r => (
                <div
                  key={r.id}
                  className="border p-2 text-sm flex gap-2 justify-between"
                >
                  <label className="flex gap-2 items-start">
                    <input
                      type="checkbox"
                      checked={ids.includes(r.id)}
                      disabled={!ids.includes(r.id) && ids.length >= 6}
                      onChange={e =>
                        setSelectedIds(
                          e.target.checked
                            ? [...ids, r.id]
                            : ids.filter(id => id !== r.id)
                        )
                      }
                    />
                    <span>
                      {date(r.startedAt)} · {r.id.slice(-6)}
                      <br />
                      <span className="text-[var(--hud-soft)]">
                        {r.samples.length} 次采样 ·{" "}
                        {r.source === "imported" ? "导入" : "现场记录"}
                      </span>
                    </span>
                  </label>
                  <button
                    className="underline shrink-0"
                    onClick={() =>
                      downloadHistory([r], "glove-calibration-round")
                    }
                  >
                    导出本轮
                  </button>
                </div>
              ))}
            </div>
          </section>
          <section className="border border-[var(--hud-line)] p-4 space-y-3">
            <div className="flex flex-wrap gap-3 items-center">
              <h2 className="font-medium">横向对比</h2>
              <label>
                动作{" "}
                <select
                  aria-label="历史动作"
                  value={axis}
                  className="border p-2 bg-[var(--hud-page)]"
                  onChange={e => {
                    setAxis(e.target.value);
                    setTarget("all");
                  }}
                >
                  <option value="all">全部动作</option>
                  {MANUAL_AXES.map(a => (
                    <option key={a} value={a}>
                      {MANUAL_MOTIONS[a].title}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                位置 / 目标{" "}
                <select
                  aria-label="历史目标"
                  value={effectiveTarget}
                  className="border p-2 bg-[var(--hud-page)]"
                  onChange={e => setTarget(e.target.value)}
                >
                  <option value="all">全部位置</option>
                  {targets.map(s => (
                    <option value={posePairKey(s)} key={posePairKey(s)}>
                      {manualTargetLabel(s.axis, s.degrees)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p className="text-sm text-[var(--hud-soft)]">
              每格为手套报告转角（相对本轮零位，0–180°），不是独立测得的真实手部角度。灰色“未采用”只表示已被重采替换或移出本轮拟合。
            </p>
            <div className="overflow-x-auto">
              <table
                className="w-full text-sm text-left border-collapse"
                aria-label="校准历史横向对比"
              >
                <thead>
                  <tr className="border-b">
                    <th className="p-3 min-w-52">你标注的实际姿势</th>
                    {selected.map(r => (
                      <th key={r.id} className="p-3 min-w-40">
                        {date(r.startedAt)}
                        <br />
                        <span className="font-normal text-xs">
                          轮次 {r.id.slice(-6)}
                        </span>
                      </th>
                    ))}
                    <th className="p-3 min-w-28">读数极差</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map(t => {
                    const all = selected.flatMap(r =>
                      r.samples
                        .filter(s => posePairKey(s) === posePairKey(t))
                        .map(s => historicalReading(r, s).measuredDeg)
                    );
                    return (
                      <tr key={posePairKey(t)} className="border-b align-top">
                        <th className="p-3 font-normal">
                          <button
                            className="underline text-left"
                            onClick={() => setTarget(posePairKey(t))}
                          >
                            {manualTargetLabel(t.axis, t.degrees)}
                          </button>
                        </th>
                        {selected.map(r => (
                          <td className="p-3" key={r.id}>
                            {r.samples
                              .filter(s => posePairKey(s) === posePairKey(t))
                              .map((s, i) => {
                                const value = historicalReading(r, s);
                                return (
                                  <div
                                    className={
                                      "mb-2 " +
                                      (!s.active
                                        ? "text-[var(--hud-soft)]"
                                        : "")
                                    }
                                    key={s.id}
                                  >
                                    <strong>
                                      {degrees(value.measuredDeg)}
                                    </strong>{" "}
                                    <span className="text-xs">
                                      第 {i + 1} 次{s.active ? "" : " · 未采用"}
                                    </span>
                                    <div className="text-xs">
                                      差值 {signed(value.differenceDeg)} ·
                                      零位后{" "}
                                      {value.elapsedSeconds === null
                                        ? "—"
                                        : value.elapsedSeconds.toFixed(1) +
                                          " 秒"}
                                    </div>
                                  </div>
                                );
                              })}
                            {!r.samples.some(
                              s => posePairKey(s) === posePairKey(t)
                            ) && "—"}
                          </td>
                        ))}
                        <td className="p-3">
                          {all.length >= 2
                            ? degrees(Math.max(...all) - Math.min(...all))
                            : "—"}
                          <div className="text-xs text-[var(--hud-soft)]">
                            {all.length} 次采样
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {!visible.length && (
              <p>所选轮次还没有该动作的采样，请记录姿势或选择其他轮次。</p>
            )}
            <p className="text-xs text-[var(--hud-soft)]">
              差值 = 手套报告转角 − 你标注的角度大小；读数极差 =
              所选轮次全部重复记录的最大值 − 最小值。点击某个姿势查看逐次记录。
            </p>
          </section>
          {effectiveTarget !== "all" && (
            <section className="border border-[var(--hud-line)] p-4 space-y-3">
              <h2 className="font-medium">这个位置的历次记录</h2>
              <p className="text-sm text-[var(--hud-soft)]">
                “与上次姿态差”同时考虑方向和转角；用于相同佩戴、相同零位动作下的比较。单看总转角相近，不代表朝向也相同。
              </p>
              <div className="overflow-x-auto">
                <table
                  className="w-full text-left text-sm"
                  aria-label="同姿势逐次记录"
                >
                  <thead>
                    <tr>
                      <th className="p-2">采样时间 / 轮次</th>
                      <th className="p-2">零位后</th>
                      <th className="p-2">手套报告</th>
                      <th className="p-2">与目标差值</th>
                      <th className="p-2">与上次读数差</th>
                      <th className="p-2">与上次姿态差</th>
                      <th className="p-2">采样波动</th>
                      <th className="p-2">状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {details.map((d, i) => {
                      const prev = details[i - 1];
                      return (
                        <tr key={d.round.id + d.sample.id} className="border-t">
                          <td className="p-2">
                            {date(d.sample.recordedAt)}
                            <br />
                            <span className="text-xs">
                              {d.round.id.slice(-6)}
                            </span>
                          </td>
                          <td className="p-2">
                            {d.elapsedSeconds === null
                              ? "—"
                              : d.elapsedSeconds.toFixed(1) + " 秒"}
                          </td>
                          <td className="p-2">{degrees(d.measuredDeg)}</td>
                          <td className="p-2">{signed(d.differenceDeg)}</td>
                          <td className="p-2">
                            {prev
                              ? signed(d.measuredDeg - prev.measuredDeg)
                              : "—"}
                          </td>
                          <td className="p-2">
                            {prev
                              ? degrees(
                                  historicalPoseDifference(
                                    prev.round,
                                    prev.sample,
                                    d.round,
                                    d.sample
                                  )
                                )
                              : "—"}
                          </td>
                          <td className="p-2">
                            {d.sample.spread === null
                              ? "—"
                              : degrees(d.sample.spread)}
                          </td>
                          <td className="p-2">
                            {d.sample.active ? "本轮采用" : "未采用，历史保留"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </main>
  );
}
