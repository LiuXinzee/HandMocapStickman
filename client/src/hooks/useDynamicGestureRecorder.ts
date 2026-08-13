import { useCallback, useEffect, useRef, useState } from "react";
import type { GloveFrame } from "@/lib/gloveProtocol";
import type { HandResult } from "@/hooks/useHandTracking";
import type {
  DynamicGestureSequence,
  DynamicGloveFrameRecord,
  DynamicVisionFrameRecord,
} from "@/lib/datasetStore";

const UI_UPDATE_INTERVAL_MS = 100;

export interface DynamicRecordingOptions {
  label: string;
  sessionId: string;
  targetDurationMs: number;
}

export interface DynamicRecorderCounts {
  left: number;
  right: number;
  vision: number;
}

export interface UseDynamicGestureRecorderReturn {
  isRecording: boolean;
  durationMs: number;
  counts: DynamicRecorderCounts;
  start: (options: DynamicRecordingOptions) => boolean;
  finish: () => DynamicGestureSequence | null;
  cancel: () => void;
  recordGloveFrame: (frame: GloveFrame) => void;
  recordVisionFrame: (result: HandResult | null) => void;
}

interface ActiveRecording extends DynamicRecordingOptions {
  startedAt: number;
  startedAtPerformance: number;
}

const emptyCounts = (): DynamicRecorderCounts => ({
  left: 0,
  right: 0,
  vision: 0,
});

export function useDynamicGestureRecorder(): UseDynamicGestureRecorderReturn {
  const mountedRef = useRef(true);
  const recordingRef = useRef(false);
  const activeRef = useRef<ActiveRecording | null>(null);
  const leftFramesRef = useRef<DynamicGloveFrameRecord[]>([]);
  const rightFramesRef = useRef<DynamicGloveFrameRecord[]>([]);
  const visionFramesRef = useRef<DynamicVisionFrameRecord[]>([]);
  const uiTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [durationMs, setDurationMs] = useState(0);
  const [counts, setCounts] = useState<DynamicRecorderCounts>(emptyCounts);

  const stopUiTimer = useCallback(() => {
    if (uiTimerRef.current) {
      clearInterval(uiTimerRef.current);
      uiTimerRef.current = null;
    }
  }, []);

  const publishUi = useCallback((durationOverride?: number) => {
    if (!mountedRef.current) return;
    const active = activeRef.current;
    const elapsed =
      durationOverride ??
      (active ? performance.now() - active.startedAtPerformance : 0);
    setDurationMs(Math.max(0, elapsed));
    setCounts({
      left: leftFramesRef.current.length,
      right: rightFramesRef.current.length,
      vision: visionFramesRef.current.length,
    });
  }, []);

  const clearBuffers = useCallback(() => {
    leftFramesRef.current = [];
    rightFramesRef.current = [];
    visionFramesRef.current = [];
  }, []);

  const start = useCallback(
    (options: DynamicRecordingOptions): boolean => {
      if (recordingRef.current || !options.label.trim()) return false;

      stopUiTimer();
      clearBuffers();
      const startedAtPerformance = performance.now();
      activeRef.current = {
        label: options.label,
        sessionId: options.sessionId,
        targetDurationMs: Math.max(0, options.targetDurationMs),
        startedAt: Date.now(),
        startedAtPerformance,
      };
      recordingRef.current = true;

      if (mountedRef.current) {
        setDurationMs(0);
        setCounts(emptyCounts());
        setIsRecording(true);
      }
      uiTimerRef.current = setInterval(publishUi, UI_UPDATE_INTERVAL_MS);
      return true;
    },
    [clearBuffers, publishUi, stopUiTimer]
  );

  const finish = useCallback((): DynamicGestureSequence | null => {
    const active = activeRef.current;
    if (!recordingRef.current || !active) return null;

    recordingRef.current = false;
    stopUiTimer();
    const finalDuration = Math.max(
      0,
      performance.now() - active.startedAtPerformance
    );
    const sequence: DynamicGestureSequence = {
      schemaVersion: 1,
      label: active.label,
      sessionId: active.sessionId,
      startedAt: active.startedAt,
      durationMs: finalDuration,
      targetDurationMs: active.targetDurationMs,
      leftFrames: leftFramesRef.current,
      rightFrames: rightFramesRef.current,
      visionFrames: visionFramesRef.current,
    };

    activeRef.current = null;
    leftFramesRef.current = [];
    rightFramesRef.current = [];
    visionFramesRef.current = [];
    if (mountedRef.current) {
      setDurationMs(finalDuration);
      setCounts({
        left: sequence.leftFrames.length,
        right: sequence.rightFrames.length,
        vision: sequence.visionFrames.length,
      });
      setIsRecording(false);
    }
    return sequence;
  }, [stopUiTimer]);

  const cancel = useCallback(() => {
    recordingRef.current = false;
    activeRef.current = null;
    stopUiTimer();
    clearBuffers();
    if (mountedRef.current) {
      setIsRecording(false);
      setDurationMs(0);
      setCounts(emptyCounts());
    }
  }, [clearBuffers, stopUiTimer]);

  const recordGloveFrame = useCallback((frame: GloveFrame) => {
    const active = activeRef.current;
    if (!recordingRef.current || !active) return;

    const record: DynamicGloveFrameRecord = {
      relativeTimeMs: Math.max(
        0,
        frame.timestamp - active.startedAtPerformance
      ),
      timestamp: frame.timestamp,
      frameId: frame.frame_id,
      hand: frame.hand,
      sensorData: [...frame.mapped_data],
      quaternion: [...frame.quaternion] as [number, number, number, number],
      acceleration: frame.acceleration ? [...frame.acceleration] : null,
      attitude: frame.attitude ? [...frame.attitude] : null,
    };
    if (frame.hand === 0x02) {
      rightFramesRef.current.push(record);
    } else {
      leftFramesRef.current.push(record);
    }
  }, []);

  const recordVisionFrame = useCallback((result: HandResult | null) => {
    const active = activeRef.current;
    if (!recordingRef.current || !active) return;

    const now = performance.now();
    visionFramesRef.current.push({
      relativeTimeMs: Math.max(0, now - active.startedAtPerformance),
      timestamp: now,
      landmarks:
        result?.landmarks.map(hand =>
          hand.map(point => ({ x: point.x, y: point.y, z: point.z }))
        ) ?? null,
      handedness: [...(result?.handedness ?? [])],
      trackingIds: [...(result?.trackingIds ?? [])],
      surfaces: [...(result?.surfaces ?? [])],
      surfaceConfidences: [...(result?.surfaceConfidences ?? [])],
      gloveConfidences: [...(result?.gloveConfidences ?? [])],
      detectionSources: [...(result?.detectionSources ?? [])],
    });
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      recordingRef.current = false;
      activeRef.current = null;
      stopUiTimer();
      clearBuffers();
    };
  }, [clearBuffers, stopUiTimer]);

  return {
    isRecording,
    durationMs,
    counts,
    start,
    finish,
    cancel,
    recordGloveFrame,
    recordVisionFrame,
  };
}
