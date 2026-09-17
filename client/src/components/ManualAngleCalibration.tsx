import { useEffect, useMemo, useRef, useState } from "react";
import { useGloveFrames } from "@/contexts/GloveContext";
import type { HandChannel } from "@/hooks/useDualGloveSerial";
import type { HandKey } from "@/lib/bendRange";
import type { GloveFrame } from "@/lib/gloveProtocol";
import {
  applyOrientationCalib,
  type OrientationCalib,
  type Quat,
} from "@/lib/orientationCalib";
import {
  AXES,
  poseAngle,
  validPose,
  type MotionAxis,
  type PosePair,
  type PoseSample,
} from "@/lib/motionCalibration";
import {
  MANUAL_AXES,
  MANUAL_MOTIONS,
  buildManualCalibration,
  captureManualPose,
  evaluateManualCalibration,
  manualTargetLabel,
  posePairKey,
  upsertPosePair,
} from "@/lib/manualOrientation";
import HandModel, { makeHandDrive } from "./HandModel";
import {
  PENDING_MANUAL_IMPORT,
  MANUAL_IMPORT_READY,
  pendingManualImport,
  readHistory,
  createRound,
  appendHistoricalSample,
  deactivateHistoricalSample,
  activePairs,
  saveHistoryRounds,
  parseHistoryImport,
  downloadHistory,
  type CalibrationRound,
} from "@/lib/calibrationHistory";

interface Props {
  hand: HandKey;
  channel: HandChannel;
  onApply: (hand: HandKey, calibration: OrientationCalib) => void;
}
const button =
  "cyber-btn px-3 py-2 rounded-sm text-sm disabled:opacity-40 aria-pressed:border-violet-500 aria-pressed:bg-violet-500/10";
const panel = "border border-[var(--hud-line)] rounded-sm p-3 space-y-3";
const IDENTITY: Quat = [1, 0, 0, 0];
const fmt = (n: number | null | undefined) =>
  n == null ? "—" : n.toFixed(1) + "°";
const targetQuaternion = (axis: MotionAxis, degrees: number): Quat => {
  const half = (degrees * Math.PI) / 360;
  return [Math.cos(half), ...AXES[axis].map(v => v * Math.sin(half))] as Quat;
};

export default function ManualAngleCalibration({
  hand,
  channel,
  onApply,
}: Props) {
  const [reference, setReference] = useState<Quat | null>(null);
  const [round, setRound] = useState<CalibrationRound | null>(null);
  const [historyError, setHistoryError] = useState("");
  const [pairs, setPairs] = useState<PosePair[]>([]);
  const [axis, setAxis] = useState<MotionAxis>("X");
  const [sign, setSign] = useState(1);
  const [angle, setAngle] = useState(30);
  const [preview, setPreview] = useState(false);
  const [previewSource, setPreviewSource] = useState<"live" | "sample">("live");
  const [confirmed, setConfirmed] = useState(false);
  const [saved, setSaved] = useState(false);
  const [message, setMessage] = useState("");
  const [live, setLive] = useState<{
    fresh: boolean;
    angle: number | null;
    error: number | null;
  }>({
    fresh: false,
    angle: null,
    error: null,
  });
  const samples = useRef<PoseSample[]>([]);
  const evidence = useRef<
    Record<string, { spread: number; recordedAt: string }>
  >({});
  const targetDrive = useRef(makeHandDrive());
  const previewDrive = useRef(makeHandDrive());
  const degrees = angle * sign;
  const targetValid = Number.isFinite(angle) && angle >= 25 && angle <= 110;
  const result = useMemo(
    () => (reference ? evaluateManualCalibration(reference, pairs) : null),
    [reference, pairs]
  );
  const candidate = useMemo<OrientationCalib | null>(
    () =>
      reference && result?.fit ? { reference, axisMap: result.fit.map } : null,
    [reference, result]
  );
  const selectedSample =
    pairs.find(p => p.axis === axis && p.degrees === degrees) ??
    pairs[0] ??
    null;
  const current = useRef({
    channel,
    reference,
    candidate,
    axis,
    degrees,
    preview,
    previewSource,
    selectedSample,
  });
  current.current = {
    channel,
    reference,
    candidate,
    axis,
    degrees,
    preview,
    previewSource,
    selectedSample,
  };
  targetDrive.current.hasData = true;
  targetDrive.current.quaternion =
    reference && targetValid ? targetQuaternion(axis, degrees) : IDENTITY;

  const ingest = (side: HandKey, frame: GloveFrame) => {
    if (side !== hand) return;
    samples.current.push({
      q:
        frame.quaternionValid === false
          ? [NaN, 0, 0, 0]
          : [...frame.quaternion],
      t: frame.timestamp,
    });
    samples.current = samples.current.filter(s => frame.timestamp - s.t <= 400);
  };
  useGloveFrames(
    f => ingest("LH", f),
    f => ingest("RH", f)
  );

  useEffect(() => {
    samples.current = [];
    // A file can be imported while disconnected. Connecting must not discard that draft.
    if (round?.source === "imported") {
      setConfirmed(false);
      if (!channel.isConnected) setPreviewSource("sample");
      return;
    }
    evidence.current = {};
    setReference(null);
    setRound(null);
    setHistoryError("");
    setPairs([]);
    setPreview(false);
    setConfirmed(false);
    setSaved(false);
    setMessage(
      channel.isConnected
        ? "连接就绪，请先摆好竖立姿势并设为零位。"
        : "请先连接这只手套。"
    );
  }, [hand, channel.isConnected]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const c = current.current;
      const frame = c.channel.latestFrameRef.current;
      const now = performance.now();
      const fresh = Boolean(
        c.channel.isConnected &&
          frame &&
          now - frame.timestamp >= 0 &&
          now - frame.timestamp < 250 &&
          frame.quaternionValid !== false &&
          validPose(frame.quaternion)
      );
      const sourceQ =
        c.previewSource === "sample"
          ? c.selectedSample?.q
          : fresh
            ? frame?.quaternion
            : null;
      previewDrive.current.hasData = Boolean(
        sourceQ && c.preview && c.candidate
      );
      const q =
        sourceQ && c.candidate
          ? applyOrientationCalib(sourceQ, c.candidate)
          : null;
      if (q && c.preview) previewDrive.current.quaternion = q;
      setLive({
        fresh,
        angle:
          fresh && frame && c.reference
            ? poseAngle(c.reference, frame.quaternion)
            : null,
        error:
          q && Number.isFinite(c.degrees)
            ? poseAngle(q, targetQuaternion(c.axis, c.degrees))
            : null,
      });
    }, 100);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!round) return;
    try {
      saveHistoryRounds([round]);
      setHistoryError("");
    } catch (e) {
      setHistoryError(
        "历史自动保存失败：" +
          (e instanceof Error ? e.message : String(e)) +
          "。本轮数据仍在，请导出 JSON 备份。"
      );
    }
  }, [round]);

  const restoreRound = (old: CalibrationRound, copy = true) => {
    const draft = {
      ...old,
      id: copy ? crypto.randomUUID() : old.id,
      source: "imported" as const,
    };
    setRound(draft);
    setReference(draft.reference);
    setPairs(activePairs(draft));
    evidence.current = {};
    if (draft.referenceRecordedAt)
      evidence.current.zero = {
        spread: 0,
        recordedAt: draft.referenceRecordedAt,
      };
    for (const sample of draft.samples.filter(s => s.active)) {
      if (sample.recordedAt)
        evidence.current[posePairKey(sample)] = {
          spread: sample.spread ?? 0,
          recordedAt: sample.recordedAt,
        };
    }
    invalidate();
    const loaded = activePairs(draft);
    const report = evaluateManualCalibration(draft.reference, loaded);
    setPreviewSource("sample");
    setPreview(Boolean(report.fit));
    if (loaded[0]) {
      setAxis(loaded[0].axis);
      setSign(Math.sign(loaded[0].degrees));
      setAngle(Math.abs(loaded[0].degrees));
    }
    setMessage(
      "已导入" +
        (hand === "RH" ? "右手" : "左手") +
        " " +
        loaded.length +
        " 个姿态。" +
        (report.fit
          ? "样本预览已开启；可切换已采姿态查看结果，连接手套后选择实时跟随。"
          : "样本已载入，暂时无法生成预览：" + report.reason) +
        "导入不会自动应用正式校准。沿用文件零位，重新上电或改变佩戴后可能不再对应。"
    );
  };
  useEffect(() => {
    const restorePending = () => {
      try {
        const pending = pendingManualImport();
        if (!pending || pending.hand !== hand) return;
        restoreRound(pending, false);
        sessionStorage.removeItem(PENDING_MANUAL_IMPORT);
        if (import.meta.env.DEV)
          void fetch("/__manus__/logs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              consoleLogs: [
                {
                  event: "manual-import-draft-restored",
                  sourceId: pending.id,
                  hand,
                  samples: activePairs(pending).length,
                },
              ],
            }),
          }).catch(() => {});
      } catch (e) {
        setMessage(
          "待导入数据读取失败：" + (e instanceof Error ? e.message : String(e))
        );
      }
    };
    restorePending();
    window.addEventListener(MANUAL_IMPORT_READY, restorePending);
    return () =>
      window.removeEventListener(MANUAL_IMPORT_READY, restorePending);
  }, [hand]);

  const importRound = async (file: File) => {
    try {
      const imported = parseHistoryImport(await file.text());
      if (imported.length !== 1)
        throw new Error("请在历史页导出单独一轮，再导入试用。");
      const old = imported[0];
      if (old.hand !== hand)
        throw new Error(
          "文件属于" +
            (old.hand === "LH" ? "左手" : "右手") +
            "，请切换手别后再导入。"
        );
      restoreRound(old);
    } catch (e) {
      setMessage("导入失败：" + (e instanceof Error ? e.message : String(e)));
    }
  };
  const restoreLatest = () => {
    try {
      const latest = readHistory()
        .filter(r => r.hand === hand && activePairs(r).length > 0)
        .at(-1);
      if (!latest) {
        setMessage("这只手还没有可恢复的采样，请先导入 JSON。");
        return;
      }
      restoreRound(latest);
    } catch (e) {
      setMessage(
        "历史读取失败：" + (e instanceof Error ? e.message : String(e))
      );
    }
  };
  const invalidate = () => {
    setPreview(false);
    setConfirmed(false);
    setSaved(false);
  };
  const capture = () => {
    if (!channel.isConnected) {
      setMessage("手套未连接，未记录。");
      return null;
    }
    const value = captureManualPose(samples.current, performance.now());
    if (!value.ok) {
      setMessage(value.reason);
      return null;
    }
    return value;
  };
  const recordZero = () => {
    const value = capture();
    if (!value) return;
    setReference(value.q);
    setRound(createRound(hand, value.q));
    setPairs([]);
    invalidate();
    evidence.current = {
      zero: { spread: value.spread, recordedAt: new Date().toISOString() },
    };
    setMessage(
      "零位已记录。选择方向与目标角度，摆好后点击“记录这个姿势”。" +
        value.reason
    );
  };
  const recordPoint = () => {
    if (!reference || !targetValid) return;
    const value = capture();
    if (!value) return;
    const pair = { axis, degrees, q: value.q };
    setPairs(old => upsertPosePair(old, pair));
    setRound(old =>
      old ? appendHistoricalSample(old, pair, value.spread) : null
    );
    invalidate();
    evidence.current[posePairKey(pair)] = {
      spread: value.spread,
      recordedAt: new Date().toISOString(),
    };
    setMessage(
      "已记录：" +
        manualTargetLabel(axis, degrees) +
        "。相同目标再次记录会替换旧数据。" +
        value.reason
    );
  };
  const removePoint = (pair: PosePair) => {
    setPairs(old => old.filter(p => posePairKey(p) !== posePairKey(pair)));
    setRound(old => (old ? deactivateHistoricalSample(old, pair) : null));
    delete evidence.current[posePairKey(pair)];
    invalidate();
    setMessage("已删除这一点，可重新摆姿势记录。");
  };
  const save = () => {
    if (
      !reference ||
      !preview ||
      previewSource !== "live" ||
      !channel.isConnected ||
      !live.fresh
    )
      return;
    const calibration = buildManualCalibration(reference, pairs, confirmed);
    if (!calibration) return;
    onApply(hand, calibration);
    setSaved(true);
    setMessage(
      (hand === "LH" ? "左手" : "右手") +
        "方向校准已应用并保存，已保留手指弯折校准。重新上电后使用“日常归零”。"
    );
  };
  const exportData = () => {
    if (round) {
      downloadHistory([round], "glove-manual-" + hand);
      return;
    }
    const url = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify(
            {
              version: 1,
              mode: "manual-angle",
              hand,
              at: new Date().toISOString(),
              reference,
              pairs,
              evidence: evidence.current,
              evaluation: result,
              visuallyConfirmed: confirmed,
              saved,
            },
            null,
            2
          ),
        ],
        { type: "application/json" }
      )
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "glove-manual-" + hand + "-" + Date.now() + ".json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="space-y-4" data-testid="manual-angle-calibration">
      <p className="text-sm text-[var(--hud-soft)]">
        由你指定实际角度，摆好后点击记录，无需摄像头。手套报告值与人工标注的差异只提示，不再因超过
        15° 阻止试用。左右手分别保存。
      </p>
      <div className="flex flex-wrap gap-2">
        <a
          className={button}
          href={"/calibration-history?hand=" + hand}
          target="_blank"
          rel="noopener noreferrer"
        >
          历史数据 / 同姿势对比 ↗
        </a>
        <label className={button}>
          导入已测 JSON 继续试用
          <input
            aria-label="导入手动校准 JSON"
            type="file"
            accept=".json,application/json"
            className="sr-only"
            onChange={e => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void importRound(file);
            }}
          />
        </label>
      </div>
      <button className={button} onClick={restoreLatest}>
        载入最近一轮{hand === "RH" ? "右手" : "左手"}历史
      </button>
      <div
        className="border border-violet-400/40 bg-violet-500/5 p-3 space-y-2"
        data-testid="calibration-import-summary"
      >
        <p className="font-medium">
          {pairs.length
            ? "已载入" +
              (hand === "RH" ? "右手" : "左手") +
              " " +
              pairs.length +
              " 个姿态"
            : "尚未载入姿态数据"}
        </p>
        <p role="status" className="text-sm">
          {message}
        </p>
        {result && (
          <p className="text-sm" data-testid="calibration-fit-summary">
            {result.reason}
          </p>
        )}
        {pairs.length > 0 && (
          <button
            className={button}
            onClick={() =>
              document
                .getElementById("calibration-preview")
                ?.scrollIntoView({ behavior: "smooth", block: "start" })
            }
          >
            查看校准预览 ↓
          </button>
        )}
      </div>
      {historyError && (
        <p role="alert" className="text-sm text-[var(--hud-err)]">
          {historyError}
        </p>
      )}
      <section className={panel} aria-label="设置零位">
        <h3 className="font-medium">① 设置零位</h3>
        <p className="text-sm">
          竖立手掌、指尖向上、手心朝自己，手指伸直。以后每个采样动作都从这个姿势出发。
        </p>
        <button
          className={button}
          disabled={!channel.isConnected || !live.fresh}
          onClick={recordZero}
        >
          {reference ? "重新设为零位（清空本轮样本）" : "设为零位"}
        </button>
        <span className="ml-3 text-sm">
          {reference ? "本轮零位已记录" : "尚未记录零位"}
        </span>
      </section>
      <section className={panel} aria-label="记录目标角度">
        <h3 className="font-medium">② 选择方向和角度，摆好后记录</h3>
        <div className="flex flex-wrap gap-2">
          {MANUAL_AXES.map(a => (
            <button
              key={a}
              className={button}
              aria-pressed={axis === a}
              onClick={() => {
                setAxis(a);
                setConfirmed(false);
              }}
            >
              {MANUAL_MOTIONS[a].title}
            </button>
          ))}
        </div>
        <p className="text-sm">{MANUAL_MOTIONS[axis].description}</p>
        <div className="flex flex-wrap gap-2">
          {[1, -1].map(s => (
            <button
              key={s}
              className={button}
              aria-pressed={sign === s}
              onClick={() => {
                setSign(s);
                setConfirmed(false);
              }}
            >
              {s > 0
                ? MANUAL_MOTIONS[axis].positive
                : MANUAL_MOTIONS[axis].negative}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          {[30, 45, 60, 90].map(d => (
            <button
              key={d}
              className={button}
              aria-pressed={angle === d}
              onClick={() => {
                setAngle(d);
                setConfirmed(false);
              }}
            >
              {d}°
            </button>
          ))}
          <label className="text-sm">
            自定角度
            <input
              className="ml-2 w-20 border bg-[var(--hud-page)] p-2"
              type="number"
              min={25}
              max={110}
              step={1}
              aria-label="自定目标角度"
              value={Number.isFinite(angle) ? angle : ""}
              onChange={e => {
                setAngle(e.target.value === "" ? NaN : Number(e.target.value));
                setConfirmed(false);
              }}
            />{" "}
            °
          </label>
        </div>
        {!targetValid && (
          <p className="text-sm text-[var(--hud-warn)]">
            请输入 25° 至 110° 的目标角度。
          </p>
        )}
        <div className="grid sm:grid-cols-2 gap-3">
          <div>
            <p className="text-sm font-medium">
              {reference
                ? "目标示意：" +
                  (targetValid
                    ? manualTargetLabel(axis, degrees)
                    : "请选择角度")
                : "零位示意：指尖向上、手心朝自己"}
            </p>
            <div className="h-56">
              <HandModel
                driveRef={targetDrive}
                side={hand === "LH" ? "left" : "right"}
                armStyle="stub"
                interactive={false}
              />
            </div>
          </div>
          <div className="space-y-3">
            <p className="text-sm">
              数据：{live.fresh ? "正在接收" : "未连接或数据中断"}
            </p>
            <p>手套报告的转角：{fmt(live.angle)}</p>
            <p className="text-sm text-[var(--hud-soft)]">
              这是相对零位的总转角，正反方向请看目标示意。记录采用点击前约 150
              毫秒数据，不等待一秒稳定。
            </p>
            <button
              className={button}
              disabled={!reference || !live.fresh || !targetValid}
              onClick={recordPoint}
            >
              记录这个姿势
              {targetValid &&
              pairs.some(p => p.axis === axis && p.degrees === degrees)
                ? "（替换同一目标）"
                : ""}
            </button>
            <p className="text-sm">
              最低需要三个动作各正反一次，共 6 个方向；建议每个方向再采
              30°、60°，共 12 个姿态。不要超出舒适活动范围。
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          {MANUAL_AXES.flatMap(a =>
            [-1, 1].map(s => {
              const done = pairs.some(
                p => p.axis === a && Math.sign(p.degrees) === s
              );
              return (
                <span
                  key={a + s}
                  className={
                    done ? "text-[var(--hud-ok)]" : "text-[var(--hud-soft)]"
                  }
                >
                  {done ? "✓" : "○"} {MANUAL_MOTIONS[a].title}{" "}
                  {s > 0 ? "正向" : "反向"}
                </span>
              );
            })
          )}
        </div>
        {pairs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr>
                  <th>你标注的实际角度</th>
                  <th>手套报告转角</th>
                  <th>与标注的差值（绝对值）</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map(pair => {
                  const row = result!.measurements.find(
                    m => m.key === posePairKey(pair)
                  )!;
                  return (
                    <tr
                      key={posePairKey(pair)}
                      className={
                        row.angleError > 15 ? "text-[var(--hud-warn)]" : ""
                      }
                    >
                      <td className="py-2">
                        {manualTargetLabel(pair.axis, pair.degrees)}
                      </td>
                      <td>{fmt(row.measuredDeg)}</td>
                      <td>{fmt(row.angleError)}</td>
                      <td>
                        <div className="flex gap-2">
                          <button
                            className={button}
                            onClick={() => {
                              setAxis(pair.axis);
                              setSign(Math.sign(pair.degrees));
                              setAngle(Math.abs(pair.degrees));
                              setConfirmed(false);
                              setMessage(
                                "已选中该目标。重新摆好实物后，点击“记录这个姿势”替换。"
                              );
                            }}
                          >
                            重采
                          </button>
                          <button
                            className={button}
                            onClick={() => removePoint(pair)}
                          >
                            删除
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <section
        id="calibration-preview"
        className={panel}
        aria-label="试用并保存"
      >
        <h3 className="font-medium">③ 生成校准，自己判断是否对应</h3>
        <button
          className={button}
          disabled={!result?.fit}
          onClick={() => {
            setPreview(true);
            setPreviewSource(live.fresh ? "live" : "sample");
            setConfirmed(false);
            setMessage(
              live.fresh
                ? "实时预览已启用。请回到零位，再做没有采过的角度和组合动作。"
                : "样本预览已启用。连接手套后可切换为实时跟随。"
            );
          }}
        >
          生成并试用校准
        </button>
        {!result?.fit && (
          <p className="text-sm text-[var(--hud-warn)]">
            {result?.reason ?? "请先导入姿态数据或记录三轴正反方向。"}
          </p>
        )}
        {preview && candidate && (
          <>
            <div className="flex flex-wrap gap-2">
              <button
                className={button}
                aria-pressed={previewSource === "sample"}
                onClick={() => {
                  setPreviewSource("sample");
                  setConfirmed(false);
                  if (selectedSample) {
                    setAxis(selectedSample.axis);
                    setSign(Math.sign(selectedSample.degrees));
                    setAngle(Math.abs(selectedSample.degrees));
                  }
                }}
              >
                查看已采样本
              </button>
              <button
                className={button}
                aria-pressed={previewSource === "live"}
                disabled={!live.fresh}
                onClick={() => {
                  setPreviewSource("live");
                  setConfirmed(false);
                }}
              >
                实时跟随手套
              </button>
            </div>
            <p className="text-sm" data-testid="preview-source">
              {previewSource === "sample"
                ? "历史样本预览：显示文件里的姿态，不随当前手掌运动。"
                : "实时手套预览：请转动手掌，检查动作是否对应。"}{" "}
              {saved ? "校准已保存。" : "尚未应用正式校准。"}
            </p>
            {!live.fresh && (
              <p className="text-sm text-[var(--hud-warn)]">
                当前没有新鲜手套数据，只能查看样本；连接后点击“实时跟随手套”。
              </p>
            )}
            {previewSource === "sample" && selectedSample && (
              <>
                <label className="block text-sm">
                  选择已采姿态{" "}
                  <select
                    aria-label="选择已采姿态"
                    className="border p-2 bg-[var(--hud-page)]"
                    value={posePairKey(selectedSample)}
                    onChange={e => {
                      const sample = pairs.find(
                        p => posePairKey(p) === e.target.value
                      );
                      if (sample) {
                        setAxis(sample.axis);
                        setSign(Math.sign(sample.degrees));
                        setAngle(Math.abs(sample.degrees));
                      }
                      setConfirmed(false);
                    }}
                  >
                    {pairs.map(p => (
                      <option key={posePairKey(p)} value={posePairKey(p)}>
                        {manualTargetLabel(p.axis, p.degrees)}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="text-sm">
                  人工标注 {Math.abs(selectedSample.degrees)}° · 手套报告{" "}
                  {fmt(poseAngle(reference!, selectedSample.q))}
                </p>
              </>
            )}
            <div className="h-64">
              <HandModel
                driveRef={previewDrive}
                side={hand === "LH" ? "left" : "right"}
                armStyle="stub"
                interactive={false}
              />
            </div>
            <p className="text-sm">
              与所选目标姿态的差距：{fmt(live.error)}
              （仅当你摆到该目标时有意义）
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                className={button}
                disabled={!live.fresh || previewSource !== "live"}
                aria-pressed={confirmed}
                onClick={() => {
                  setConfirmed(true);
                  setMessage("已确认预览对应，可以保存到这只手。");
                }}
              >
                我看着对应
              </button>
              <button
                className={button}
                onClick={() => {
                  setConfirmed(false);
                  setPreview(false);
                  setMessage(
                    "请核对目标方向，并在表格中重采或删除有疑问的姿态，再生成校准。"
                  );
                }}
              >
                方向还不对，返回调整
              </button>
              <button
                className={button}
                disabled={
                  !confirmed || previewSource !== "live" || !live.fresh || saved
                }
                onClick={save}
              >
                {saved
                  ? "已保存"
                  : "保存" + (hand === "LH" ? "左手" : "右手") + "校准"}
              </button>
            </div>
          </>
        )}
        <p className="text-xs text-[var(--hud-soft)]">
          大于 15°
          的角度差和转轴偏差都只提示，可先试用再自行决定是否保存。本流程仍只对齐方向，不缩放角度、不处理漂移；无法计算有效映射时仍不能生成。修改样本后需要重新生成并确认。
        </p>
      </section>
      <button className={button} disabled={!reference} onClick={exportData}>
        导出本轮手动校准 JSON
      </button>
      <p className="text-xs text-[var(--hud-soft)]">
        每次记录自动保存到历史，重采和删除旧点也保留历史读数。关闭、换手或断线会结束当前草稿；可在历史页查看并导出单轮
        JSON，导入后继续试用。正式校准仍需点击保存。
      </p>
    </div>
  );
}
