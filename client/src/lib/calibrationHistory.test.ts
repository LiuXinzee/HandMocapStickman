import { describe, expect, it } from "vitest";
import {
  HISTORY_KEY,
  importHistoryRounds,
  createRound,
  appendHistoricalSample,
  deactivateHistoricalSample,
  readHistory,
  saveHistoryRounds,
  historyJSON,
  parseHistoryImport,
  historicalReading,
  historicalPoseDifference,
  activePairs,
} from "./calibrationHistory";
import { type Quat } from "./orientationCalib";
import { AXES, type PosePair } from "./motionCalibration";

const I: Quat = [1, 0, 0, 0];
const at = (seconds: number) =>
  new Date(Date.UTC(2026, 8, 17, 0, 0, seconds)).toISOString();
const pair = (degrees: number, measured = degrees): PosePair => ({
  axis: "X",
  degrees,
  q: [
    Math.cos((measured * Math.PI) / 360),
    Math.sin((measured * Math.PI) / 360),
    0,
    0,
  ],
});
function memory() {
  const values = new Map<string, string>();
  return {
    getItem: (k: string) => values.get(k) ?? null,
    setItem: (k: string, v: string) => {
      values.set(k, v);
    },
  };
}
describe("calibration sample history", () => {
  it("retains every replacement and deletion while fitting only active samples", () => {
    const empty = createRound("RH", I, at(0));
    const first = appendHistoricalSample(empty, pair(30, 7.2), 0.2, at(10));
    const second = appendHistoricalSample(first, pair(30, 20), 0.3, at(30));
    expect(first.samples[0].active).toBe(true);
    expect(second.samples.map(s => s.active)).toEqual([false, true]);
    expect(activePairs(second)).toEqual([pair(30, 20)]);
    const removed = deactivateHistoricalSample(second, pair(30));
    expect(removed.samples).toHaveLength(2);
    expect(activePairs(removed)).toEqual([]);
    const reading = historicalReading(second, second.samples[0]);
    expect(reading.measuredDeg).toBeCloseTo(7.2);
    expect(reading.differenceDeg).toBeCloseTo(-22.8);
    expect(reading.elapsedSeconds).toBe(10);
    expect(
      historicalPoseDifference(
        second,
        second.samples[0],
        second,
        second.samples[1]
      )
    ).toBeCloseTo(12.8);
  });
  it("persists across reloads, merges rounds without mixing hands, and never silently evicts earlier samples", () => {
    const storage = memory();
    const r = createRound("RH", I, at(0)),
      l = createRound("LH", I, at(1));
    saveHistoryRounds([r], storage);
    saveHistoryRounds([l], storage);
    saveHistoryRounds(
      [appendHistoricalSample(r, pair(60, 170), 1, at(20))],
      storage
    );
    const read = readHistory(storage);
    expect(read).toHaveLength(2);
    expect(read.find(x => x.id === r.id)!.samples).toHaveLength(1);
    expect(read.find(x => x.id === l.id)!.samples).toHaveLength(0);
    const imported = parseHistoryImport(historyJSON(read));
    saveHistoryRounds(imported, storage);
    expect(readHistory(storage)).toEqual(read);
  });
  it("imports legacy manual samples with original zero/time and rejects other diagnosis files", () => {
    const file = {
      version: 1,
      mode: "manual-angle",
      hand: "RH",
      at: at(50),
      reference: I,
      pairs: [pair(60, 175.3)],
      evidence: {
        zero: { recordedAt: at(0), spread: 0 },
        "X:60": { recordedAt: at(40), spread: 1.2 },
      },
    };
    const imported = parseHistoryImport(JSON.stringify(file))[0];
    expect(imported.source).toBe("imported");
    expect(
      historicalReading(imported, imported.samples[0]).elapsedSeconds
    ).toBe(40);
    expect(
      historicalReading(imported, imported.samples[0]).measuredDeg
    ).toBeCloseTo(175.3);
    expect(parseHistoryImport(JSON.stringify(file))[0].id).toBe(imported.id);
    expect(() =>
      parseHistoryImport(JSON.stringify({ ...file, mode: "still" }))
    ).toThrow();
    expect(() =>
      parseHistoryImport(JSON.stringify({ ...file, reference: [0, 0, 0, 0] }))
    ).toThrow();
    expect(() =>
      parseHistoryImport(
        JSON.stringify({ ...file, pairs: [{ ...pair(30), degrees: NaN }] })
      )
    ).toThrow();
    const unknown = parseHistoryImport(
      JSON.stringify({ ...file, evidence: {} })
    )[0];
    expect(
      historicalReading(unknown, unknown.samples[0]).elapsedSeconds
    ).toBeNull();
    expect(unknown.samples[0].recordedAt).toBeNull();
  });
  it("keeps malformed stored data intact and surfaces quota errors", () => {
    const storage = memory();
    storage.setItem(HISTORY_KEY, "broken json");
    expect(() =>
      saveHistoryRounds([createRound("RH", I, at(0))], storage)
    ).toThrow();
    expect(storage.getItem(HISTORY_KEY)).toBe("broken json");
    const quota = {
      getItem: () => null,
      setItem: () => {
        throw new Error("QuotaExceededError");
      },
    };
    expect(() =>
      saveHistoryRounds([createRound("RH", I, at(0))], quota)
    ).toThrow("QuotaExceededError");
    expect(() =>
      parseHistoryImport(
        '{"kind":"manual-calibration-history","version":9,"rounds":[]}'
      )
    ).toThrow();
  });
  it("detects different directions with identical total angles, ignoring quaternion sign", () => {
    const round = createRound("RH", I, at(0));
    const first = appendHistoricalSample(round, pair(60), 0, at(1));
    const sample = first.samples[0];
    const y = {
      ...sample,
      q: [Math.cos(Math.PI / 6), 0, Math.sin(Math.PI / 6), 0] as Quat,
    };
    expect(historicalReading(round, y).measuredDeg).toBeCloseTo(60);
    expect(historicalPoseDifference(round, sample, round, y)).toBeGreaterThan(
      80
    );
    expect(
      historicalPoseDifference(round, sample, round, {
        ...sample,
        q: sample.q.map(x => -x) as Quat,
      })
    ).toBeLessThan(0.001);
  });
});

it("importing an old backup preserves the newer round and is idempotent", () => {
  const storage = memory();
  const first = appendHistoricalSample(
    createRound("RH", I, at(0)),
    pair(30, 7),
    0,
    at(1)
  );
  const second = appendHistoricalSample(first, pair(30, 17), 0, at(10));
  saveHistoryRounds([second], storage);
  const imported = importHistoryRounds([first], storage);
  expect(imported[0].id).not.toBe(second.id);
  expect(
    readHistory(storage).find(r => r.id === second.id)!.samples
  ).toHaveLength(2);
  importHistoryRounds([first], storage);
  expect(readHistory(storage)).toHaveLength(2);
});
