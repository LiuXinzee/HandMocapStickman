import type { HandKey } from "./bendRange";
import { poseAngle, validPose, type PosePair } from "./motionCalibration";
import {
  conjugateQuat,
  multiplyQuat,
  normalizeQuat,
  type Quat,
} from "./orientationCalib";

export const HISTORY_KEY = "deafkit_manual_calibration_history_v1";
export const HISTORY_EVENT = "manual-calibration-history-change";
export interface HistoricalSample extends PosePair {
  id: string;
  recordedAt: string | null;
  spread: number | null;
  active: boolean;
}
export interface CalibrationRound {
  id: string;
  hand: HandKey;
  startedAt: string;
  referenceRecordedAt: string | null;
  reference: Quat;
  source: "live" | "imported";
  samples: HistoricalSample[];
}
type StorageLike = Pick<Storage, "getItem" | "setItem">;
const date = (v: unknown): v is string =>
  typeof v === "string" && Number.isFinite(Date.parse(v));
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const quaternion = (v: unknown): v is Quat =>
  Array.isArray(v) &&
  v.length === 4 &&
  v.every(x => typeof x === "number") &&
  validPose(v as Quat);
function isPair(v: unknown): v is PosePair {
  return (
    object(v) &&
    ["X", "Y", "Z"].includes(v.axis as string) &&
    typeof v.degrees === "number" &&
    Number.isFinite(v.degrees) &&
    Math.abs(v.degrees) >= 25 &&
    Math.abs(v.degrees) <= 110 &&
    quaternion(v.q)
  );
}
function isRound(v: unknown): v is CalibrationRound {
  return (
    object(v) &&
    typeof v.id === "string" &&
    !!v.id &&
    (v.hand === "LH" || v.hand === "RH") &&
    date(v.startedAt) &&
    (v.referenceRecordedAt === null || date(v.referenceRecordedAt)) &&
    quaternion(v.reference) &&
    (v.source === "live" || v.source === "imported") &&
    Array.isArray(v.samples) &&
    v.samples.every(
      s =>
        isPair(s) &&
        object(s) &&
        typeof s.id === "string" &&
        !!s.id &&
        (s.recordedAt === null || date(s.recordedAt)) &&
        (s.spread === null ||
          (typeof s.spread === "number" &&
            Number.isFinite(s.spread) &&
            s.spread >= 0)) &&
        typeof s.active === "boolean"
    ) &&
    new Set(v.samples.map(s => s.id)).size === v.samples.length
  );
}
export function createRound(
  hand: HandKey,
  reference: Quat,
  at = new Date().toISOString()
): CalibrationRound {
  return {
    id: crypto.randomUUID(),
    hand,
    reference: normalizeQuat(reference),
    startedAt: at,
    referenceRecordedAt: at,
    source: "live",
    samples: [],
  };
}
export function appendHistoricalSample(
  round: CalibrationRound,
  pair: PosePair,
  spread: number,
  at = new Date().toISOString()
): CalibrationRound {
  return {
    ...round,
    samples: [
      ...round.samples.map(s =>
        s.axis === pair.axis && s.degrees === pair.degrees
          ? { ...s, active: false }
          : s
      ),
      {
        ...pair,
        q: [...pair.q],
        id: crypto.randomUUID(),
        spread,
        recordedAt: at,
        active: true,
      },
    ],
  };
}
export function deactivateHistoricalSample(
  round: CalibrationRound,
  pair: PosePair
): CalibrationRound {
  return {
    ...round,
    samples: round.samples.map(s =>
      s.axis === pair.axis && s.degrees === pair.degrees
        ? { ...s, active: false }
        : s
    ),
  };
}
export function activePairs(round: CalibrationRound): PosePair[] {
  return round.samples
    .filter(s => s.active)
    .map(s => ({ axis: s.axis, degrees: s.degrees, q: [...s.q] }));
}
export function historicalReading(
  round: CalibrationRound,
  sample: HistoricalSample
) {
  const measuredDeg = poseAngle(round.reference, sample.q);
  return {
    measuredDeg,
    differenceDeg: measuredDeg - Math.abs(sample.degrees),
    elapsedSeconds:
      round.referenceRecordedAt && sample.recordedAt
        ? Math.max(
            0,
            (Date.parse(sample.recordedAt) -
              Date.parse(round.referenceRecordedAt)) /
              1000
          )
        : null,
  };
}
/** Compare relative orientations, not raw power-on world headings. q and -q are equivalent. */
export function historicalPoseDifference(
  a: CalibrationRound,
  sa: HistoricalSample,
  b: CalibrationRound,
  sb: HistoricalSample
) {
  return poseAngle(
    multiplyQuat(
      conjugateQuat(normalizeQuat(a.reference)),
      normalizeQuat(sa.q)
    ),
    multiplyQuat(conjugateQuat(normalizeQuat(b.reference)), normalizeQuat(sb.q))
  );
}
function readEnvelope(value: unknown): CalibrationRound[] {
  if (
    !object(value) ||
    value.version !== 1 ||
    value.kind !== "manual-calibration-history" ||
    !Array.isArray(value.rounds) ||
    !value.rounds.every(isRound) ||
    new Set(value.rounds.map(r => r.id)).size !== value.rounds.length
  )
    throw new Error("历史文件格式无效，未覆盖已有记录。");
  return value.rounds;
}
export function historyJSON(rounds: CalibrationRound[]) {
  return JSON.stringify(
    { version: 1, kind: "manual-calibration-history", rounds },
    null,
    2
  );
}
export function readHistory(
  storage: StorageLike = localStorage
): CalibrationRound[] {
  const raw = storage.getItem(HISTORY_KEY);
  if (!raw) return [];
  return readEnvelope(JSON.parse(raw));
}
export function saveHistoryRounds(
  incoming: CalibrationRound[],
  storage: StorageLike = localStorage
) {
  if (!incoming.every(isRound)) throw new Error("采样历史格式无效，未保存。");
  // Read afresh so another tab's rounds are retained. Never silently truncate old samples.
  const rounds = readHistory(storage);
  for (const round of incoming) {
    const index = rounds.findIndex(r => r.id === round.id);
    if (index < 0) rounds.push(round);
    else rounds[index] = round;
  }
  storage.setItem(HISTORY_KEY, historyJSON(rounds));
  if (typeof window !== "undefined")
    window.dispatchEvent(new Event(HISTORY_EVENT));
}
/** Importing an older backup must never overwrite later measurements. */
export function importHistoryRounds(
  incoming: CalibrationRound[],
  storage: StorageLike = localStorage
) {
  const existing = readHistory(storage);
  const preserved = incoming.map(round => {
    const old = existing.find(r => r.id === round.id);
    if (!old || JSON.stringify(old) === JSON.stringify(round)) return round;
    return {
      ...round,
      id: round.id + "-copy-" + legacyId(JSON.stringify(round)).slice(7),
    };
  });
  saveHistoryRounds(preserved, storage);
  return preserved;
}
function legacyId(text: string) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++)
    hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
  return "legacy-" + (hash >>> 0).toString(16);
}
/** Also reads the manual-angle JSON exported before automatic history was introduced. */
export function parseHistoryImport(text: string): CalibrationRound[] {
  const data: unknown = JSON.parse(text);
  if (object(data) && data.kind === "manual-calibration-history")
    return readEnvelope(data);
  if (
    !object(data) ||
    data.mode !== "manual-angle" ||
    data.version !== 1 ||
    (data.hand !== "LH" && data.hand !== "RH") ||
    !quaternion(data.reference) ||
    !Array.isArray(data.pairs) ||
    !data.pairs.every(isPair)
  )
    throw new Error(
      "请选择手动校准 JSON 或校准历史 JSON，未导入其他诊断格式。"
    );
  const evidence = object(data.evidence) ? data.evidence : {};
  const zero = object(evidence.zero) ? evidence.zero : {};
  const referenceAt = date(zero.recordedAt) ? zero.recordedAt : null;
  const id = legacyId(
    JSON.stringify([data.hand, data.at, data.reference, data.pairs, evidence])
  );
  const samples: HistoricalSample[] = data.pairs.map((p, i) => {
    const entry = evidence[p.axis + ":" + p.degrees];
    const e = object(entry) ? entry : {};
    return {
      ...p,
      id: id + "-" + i,
      q: normalizeQuat(p.q),
      active: true,
      recordedAt: date(e.recordedAt) ? e.recordedAt : null,
      spread:
        typeof e.spread === "number" &&
        Number.isFinite(e.spread) &&
        e.spread >= 0
          ? e.spread
          : null,
    };
  });
  if (
    new Set(samples.map(s => s.axis + ":" + s.degrees)).size !== samples.length
  )
    throw new Error("文件包含重复目标，无法确定本轮采用的点位。");
  return [
    {
      id,
      hand: data.hand,
      source: "imported",
      startedAt:
        referenceAt ?? (date(data.at) ? data.at : new Date().toISOString()),
      referenceRecordedAt: referenceAt,
      reference: normalizeQuat(data.reference),
      samples,
    },
  ];
}
export function downloadHistory(
  rounds: CalibrationRound[],
  name = "glove-calibration-history"
) {
  const url = URL.createObjectURL(
    new Blob([historyJSON(rounds)], { type: "application/json" })
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name + "-" + Date.now() + ".json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A staged import is a draft only; it never writes the applied orientation parameters. */
export const PENDING_MANUAL_IMPORT = "deafkit_pending_manual_import_v1";
export const MANUAL_IMPORT_READY = "manual-calibration-import-ready";
export function pendingManualImport(): CalibrationRound | null {
  try {
    const text = sessionStorage.getItem(PENDING_MANUAL_IMPORT);
    if (!text) return null;
    const rounds = parseHistoryImport(text);
    return rounds.length === 1 ? rounds[0] : null;
  } catch {
    return null;
  }
}
