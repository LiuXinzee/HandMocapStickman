import { useEffect, useRef, useState, type RefObject } from "react";
import { useGloveFrames } from "@/contexts/GloveContext";
import type { HandChannel } from "@/hooks/useDualGloveSerial";
import type { GloveFrame } from "@/lib/gloveProtocol";
import type { HandKey } from "@/lib/bendRange";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import ManualAngleCalibration from "./ManualAngleCalibration";
import {
  MANUAL_IMPORT_READY,
  pendingManualImport,
} from "@/lib/calibrationHistory";
import HandModel, { makeHandDrive, type HandDrive } from "./HandModel";
import {
  applyOrientationCalib,
  type OrientationCalib,
  type Quat,
} from "@/lib/orientationCalib";
import {
  AXES,
  checkCaptureStart,
  pairError,
  poseAngle,
  rebaseCalibration,
  stationaryReport,
  validPose,
  type MotionAxis,
  type PosePair,
  type PoseSample,
} from "@/lib/motionCalibration";

type Mode = "zero" | "still" | "verify" | "fit";
type CaptureKind = "zero" | "still" | "baseline" | "point";
interface Reading extends PosePair {
  rawDeg: number;
  calibratedDeg: number;
  shownDeg: number | null;
  axisError: number | null;
}
interface Props {
  initialMode?: "zero" | "fit";
  channels: Record<HandKey, HandChannel>;
  calibrations: Record<HandKey, OrientationCalib | null>;
  drives: Record<HandKey, RefObject<HandDrive>>;
  onApply: (hand: HandKey, calib: OrientationCalib) => void;
  onClose: () => void;
}
const button =
  "cyber-btn px-3 py-2 rounded-sm text-sm disabled:opacity-40 aria-pressed:border-violet-500 aria-pressed:bg-violet-500/10";
const modes: Record<Mode, string> = {
  zero: "日常归零",
  still: "静置检测",
  verify: "三轴验证",
  fit: "手动角度校准",
};
const axisText: Record<MotionAxis, string> = {
  X: "左右水平轴：竖立手向前或向后倾",
  Y: "上下竖直轴：掌心朝向左右转",
  Z: "前后水平轴：竖立手向左或向右侧倾",
};
const fmt = (value: number | null | undefined) =>
  value == null ? "—" : `${value.toFixed(1)}°`;

export default function MotionDiagnostics(props: Props) {
  const [hand, setHand] = useState<HandKey>(
    pendingManualImport()?.hand ?? (props.channels.RH.isConnected ? "RH" : "LH")
  );
  const [mode, setMode] = useState<Mode>(props.initialMode ?? "zero");
  useEffect(() => {
    const imported = () => {
      const pending = pendingManualImport();
      if (!pending) return;
      reset();
      setHand(pending.hand);
      setMode("fit");
    };
    window.addEventListener(MANUAL_IMPORT_READY, imported);
    return () => window.removeEventListener(MANUAL_IMPORT_READY, imported);
  }, []);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const [axis, setAxis] = useState<MotionAxis>("Y");
  const [degrees, setDegrees] = useState(60);
  const [baseline, setBaseline] = useState<{
    q: Quat;
    shown: Quat | null;
  } | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [report, setReport] =
    useState<ReturnType<typeof stationaryReport>>(null);
  const [live, setLive] = useState<{
    raw: number;
    calibrated: number;
    shown: number | null;
  } | null>(null);
  const history = useRef<PoseSample[]>([]);
  const lastRecording = useRef<PoseSample[]>([]);
  const pending = useRef<{
    kind: CaptureKind;
    start: number;
    recordingAt: number | null;
    samples: PoseSample[];
    axis: MotionAxis;
    degrees: number;
  } | null>(null);
  const current = useRef({ props, hand, baseline, axis, degrees });
  current.current = { props, hand, baseline, axis, degrees };
  const demo = useRef<HandDrive>(makeHandDrive());
  demo.current.hasData = true;
  const h = (degrees * Math.PI) / 360;
  demo.current.quaternion =
    mode === "verify"
      ? ([Math.cos(h), ...AXES[axis].map(v => v * Math.sin(h))] as Quat)
      : [1, 0, 0, 0];
  const ingest = (side: HandKey, frame: GloveFrame) => {
    if (side !== current.current.hand) return;
    const s: PoseSample = {
      q: frame.quaternionValid === false ? [NaN, 0, 0, 0] : frame.quaternion,
      t: frame.timestamp,
    };
    history.current.push(s);
    history.current = history.current.filter(x => s.t - x.t <= 1400);
    if (pending.current?.recordingAt != null) pending.current.samples.push(s);
  };
  useGloveFrames(
    f => ingest("LH", f),
    f => ingest("RH", f)
  );
  const cancel = () => {
    pending.current = null;
    setBusy(false);
    setProgress(0);
  };
  const reset = () => {
    cancel();
    history.current = [];
    lastRecording.current = [];
    setBaseline(null);
    setReadings([]);
    setReport(null);
    setMessage("");
    setLive(null);
  };
  const start = (
    kind: CaptureKind,
    selectedAxis = axis,
    selectedDegrees = degrees
  ) => {
    history.current = [];
    if (kind === "still") {
      setReport(null);
      lastRecording.current = [];
    }
    if (kind === "baseline") {
      setBaseline(null);
      setReadings([]);
    }
    pending.current = {
      kind,
      start: performance.now(),
      recordingAt: null,
      samples: [],
      axis: selectedAxis,
      degrees: selectedDegrees,
    };
    setBusy(true);
    setProgress(0);
    setMessage(
      kind === "still"
        ? "请固定实物，3 秒后开始记录；不要求角度读数稳定。"
        : "保持姿态不动，正在等待稳定数据…"
    );
  };

  useEffect(() => {
    const timer = window.setInterval(() => {
      const { props: p, hand: side, baseline: base } = current.current;
      const frame = p.channels[side].latestFrameRef.current;
      const now = performance.now();
      const fresh =
        p.channels[side].isConnected &&
        frame &&
        now - frame.timestamp < 250 &&
        frame.quaternionValid !== false &&
        validPose(frame.quaternion);
      if (fresh && base) {
        const calib = p.calibrations[side];
        const rendered = p.drives[side].current.renderedQuaternion;
        setLive({
          raw: poseAngle(base.q, frame.quaternion),
          calibrated: poseAngle(
            applyOrientationCalib(base.q, calib),
            applyOrientationCalib(frame.quaternion, calib)
          ),
          shown:
            base.shown && rendered ? poseAngle(base.shown, rendered) : null,
        });
      } else setLive(null);
      const task = pending.current;
      if (!task) return;
      if (!p.channels[side].isConnected) {
        cancel();
        setMessage("连接已断开，检测取消。重连后请重新记录起点。");
        setBaseline(null);
        return;
      }
      if (task.recordingAt === null) {
        const gate = checkCaptureStart(
          task.kind === "still" ? "still" : "pose",
          history.current,
          task.start,
          now
        );
        if (gate.ready) {
          if (task.kind === "still") {
            task.recordingAt = now;
            task.samples = [];
            setMessage("保持实物不动，正在记录 10 秒；角度变化也会完整记录。");
            return;
          }
          if (!gate.mean) return;
          const reference = gate.mean;
          const rendered = p.drives[side].current.renderedQuaternion;
          lastRecording.current = [...history.current];
          if (task.kind === "zero") {
            p.onApply(side, rebaseCalibration(p.calibrations[side], reference));
            setMessage(
              p.calibrations[side]?.axisMap
                ? "零位已更新，已保留安装方向与手指弯折校准。"
                : "已记录零位；尚无安装方向校准，请使用手动角度校准。"
            );
          } else if (task.kind === "baseline") {
            setBaseline({
              q: reference,
              shown: rendered ? [...rendered] : null,
            });
            setReadings([]);
            setMessage(
              "参考姿态已记录。每次从同一参考姿态出发，摆到目标后再记录。"
            );
          } else if (base) {
            const pair: PosePair = {
              axis: task.axis,
              degrees: task.degrees,
              q: reference,
            };
            if (task.kind === "point") {
              const calib = p.calibrations[side];
              const errors = calib?.axisMap
                ? pairError(base.q, calib.axisMap, pair)
                : null;
              setReadings(old => [
                ...old,
                {
                  ...pair,
                  rawDeg: poseAngle(base.q, reference),
                  calibratedDeg: poseAngle(
                    applyOrientationCalib(base.q, calib),
                    applyOrientationCalib(reference, calib)
                  ),
                  shownDeg:
                    base.shown && rendered
                      ? poseAngle(base.shown, rendered)
                      : null,
                  axisError: errors?.axisError ?? null,
                },
              ]);
              setMessage(
                "测量已记录。目标角度需要用实物或角度线确认，软件无法自动知道真实角度。"
              );
            }
          }
          cancel();
          return;
        }
        setMessage(
          task.kind === "still"
            ? gate.reason
            : gate.reason + "；摆好后保持约 1 秒。"
        );
        if (now - task.start > 15000) {
          cancel();
          setMessage(
            "15 秒内未取得稳定姿态，未写入。请检查连接、手套固定和数据，再重试。"
          );
        }
      } else {
        const elapsed = now - task.recordingAt;
        setProgress(Math.min(100, elapsed / 100));
        if (elapsed >= 10000) {
          lastRecording.current = [...task.samples];
          const result = stationaryReport(task.samples);
          setReport(result);
          cancel();
          setMessage(
            !result ||
              result.seconds < 9 ||
              result.gaps > 0 ||
              result.invalid > 0 ||
              !frame ||
              now - frame.timestamp > 250
              ? "数据不足或中断，请重测。此结果不能用于判断静置表现。"
              : "已记录。请确认这 10 秒内实物没有移动；累计变化不等于持续漂移。"
          );
        }
      }
    }, 100);
    return () => {
      window.clearInterval(timer);
      pending.current = null;
    };
  }, []);
  const connected = props.channels[hand].isConnected;
  useEffect(() => {
    reset();
    setMessage(
      connected
        ? "连接就绪。重连后请重新记录本轮参考姿态。"
        : "尚未连接手套，可先查看操作说明。"
    );
  }, [hand, connected]);

  const exportData = () => {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            version: 1,
            at: new Date().toISOString(),
            hand,
            mode,
            calibration: props.calibrations[hand],
            baseline,
            report,
            readings,
            samples: lastRecording.current,
          },
          null,
          2
        ),
      ],
      { type: "application/json" }
    );
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = `glove-motion-${hand}-${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <Dialog
      open
      onOpenChange={open => {
        if (!open) props.onClose();
      }}
    >
      <DialogContent
        showCloseButton={false}
        onInteractOutside={e => e.preventDefault()}
        className="motion-calibration sm:max-w-4xl max-h-[calc(100dvh-2rem)] overflow-y-auto bg-[var(--hud-page)] border border-[var(--hud-line)] rounded-sm p-5 gap-4 text-[var(--hud-text)]"
      >
        <div className="flex items-center justify-between gap-3">
          <DialogTitle className="font-medium">朝向检查与校准</DialogTitle>
          <button className={button} onClick={props.onClose}>
            关闭
          </button>
        </div>
        <DialogDescription className="text-sm text-[var(--hud-soft)]">
          手动角度校准用于对齐转动方向；日常重新上电后快速归零。手指弯折可继续使用四步校准。
        </DialogDescription>
        <div className="flex flex-wrap gap-2">
          {(["LH", "RH"] as const).map(side => (
            <button
              key={side}
              className={button}
              aria-pressed={hand === side}
              disabled={busy}
              onClick={() => {
                reset();
                setHand(side);
              }}
            >
              {side === "LH" ? "左手" : "右手"} ·{" "}
              {props.channels[side].isConnected ? "已连接" : "未连接"}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {(Object.keys(modes) as Mode[]).map(m => (
            <button
              key={m}
              className={button}
              aria-pressed={mode === m}
              disabled={busy}
              onClick={() => {
                reset();
                setMode(m);
              }}
            >
              {modes[m]}
            </button>
          ))}
        </div>
        {mode === "zero" && (
          <>
            <p>
              竖立手掌、指尖向上、手心朝自己，保持不动。只更新本次使用的零位，保留已有安装方向与弯折量程。
            </p>
            <button
              className={button}
              disabled={busy || !connected}
              onClick={() => start("zero")}
            >
              稳定后更新零位
            </button>
          </>
        )}
        {mode === "still" && (
          <>
            <p>
              把手套平放在桌面，或把前臂支撑好、手掌竖立并扶稳。点击后准备 3
              秒，再记录 10
              秒；期间不要移动实物，不需要等角度读数稳定。这里只测姿态变化，不能单凭结果判定硬件损坏。
            </p>
            <button
              className={button}
              disabled={busy || !connected}
              onClick={() => start("still")}
            >
              开始 10 秒静置检测
            </button>
            {report && (
              <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                <div>
                  <dt>首尾平均姿态差</dt>
                  <dd>{fmt(report.netDeg)}</dd>
                </div>
                <div>
                  <dt>趋势外波动 P95</dt>
                  <dd>{fmt(report.residualP95Deg)}</dd>
                </div>
                <div>
                  <dt>累计变化（不是零偏）</dt>
                  <dd>{report.pathDegPerMin.toFixed(0)}°/分钟</dd>
                </div>
                <div>
                  <dt>突跳 / 中断 / 无效帧</dt>
                  <dd>
                    {report.jumps} / {report.gaps} / {report.invalid}
                  </dd>
                </div>
                <div>
                  <dt>实际记录时长</dt>
                  <dd>{report.seconds.toFixed(1)} 秒</dd>
                </div>
              </dl>
            )}
          </>
        )}
        {mode === "fit" && (
          <ManualAngleCalibration
            key={hand}
            hand={hand}
            channel={props.channels[hand]}
            onApply={props.onApply}
          />
        )}
        {mode === "verify" && (
          <>
            <p>
              先竖立、指尖向上、手心朝自己记录参考。每个目标都从这个姿态出发，照右侧示意转动；X
              左右、Y 上下、Z
              朝向自己。保持浏览器视角默认，目标角度请用实物角度线核对。
            </p>
            <button
              className={button}
              disabled={busy || !connected}
              onClick={() => start("baseline")}
            >
              {baseline ? "重新记录参考（清空本轮测量）" : "记录竖立参考姿态"}
            </button>
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-3">
                <label className="block">
                  转轴{" "}
                  <select
                    className="ml-2 p-2 bg-[var(--hud-page)] border"
                    value={axis}
                    disabled={busy}
                    onChange={e => setAxis(e.target.value as MotionAxis)}
                  >
                    {Object.keys(AXES).map(a => (
                      <option key={a}>{a}</option>
                    ))}
                  </select>
                </label>
                <p className="text-sm">{axisText[axis]}</p>
                <label className="block">
                  目标角度{" "}
                  <select
                    className="ml-2 p-2 bg-[var(--hud-page)] border"
                    value={degrees}
                    disabled={busy}
                    onChange={e => setDegrees(Number(e.target.value))}
                  >
                    {[-90, -60, -45, -30, 30, 45, 60, 90].map(d => (
                      <option value={d} key={d}>
                        {d}°
                      </option>
                    ))}
                  </select>
                </label>
                <p className="text-sm">
                  原始转角 {fmt(live?.raw)} · 校准后 {fmt(live?.calibrated)} ·
                  实际渲染 {fmt(live?.shown)}
                </p>
                <button
                  className={button}
                  disabled={busy || !baseline || !connected}
                  onClick={() => start("point")}
                >
                  记录当前目标
                </button>
              </div>
              <div>
                <p className="text-sm">
                  目标示意（预设姿态） · {axis} {degrees}°
                </p>
                <div className="h-64">
                  <HandModel
                    driveRef={demo}
                    side={hand === "LH" ? "left" : "right"}
                    armStyle="stub"
                  />
                </div>
              </div>
            </div>
            {readings.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead>
                    <tr>
                      <th>目标</th>
                      <th>原始</th>
                      <th>校准后</th>
                      <th>渲染后</th>
                      <th>转轴偏差</th>
                    </tr>
                  </thead>
                  <tbody>
                    {readings.map((r, i) => (
                      <tr key={i}>
                        <td>
                          {r.axis} {r.degrees}°
                        </td>
                        <td>{fmt(r.rawDeg)}</td>
                        <td>{fmt(r.calibratedDeg)}</td>
                        <td>{fmt(r.shownDeg)}</td>
                        <td>{fmt(r.axisError)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
        {mode !== "fit" && (
          <>
            {busy && (
              <div className="flex items-center gap-3">
                <progress className="flex-1" value={progress} max={100} />
                <button
                  className={button}
                  onClick={() => {
                    cancel();
                    setMessage("已取消，未写入。");
                  }}
                >
                  取消测量
                </button>
              </div>
            )}
            <p role="status" className="text-sm text-[var(--hud-soft)]">
              {message}
            </p>
            <button
              className={button}
              disabled={
                busy ||
                (!report && !readings.length && !lastRecording.current.length)
              }
              onClick={exportData}
            >
              导出本轮诊断 JSON
            </button>
          </>
        )}
        <p className="text-xs text-[var(--hud-soft)]">
          校准保存在当前浏览器、当前地址下，左右手分别保存；请固定使用同一端口。检测结果只代表本次采样，不是永久健康证明。
        </p>
      </DialogContent>
    </Dialog>
  );
}
