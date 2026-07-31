import { describe, expect, it, vi } from "vitest";
import {
  ConcurrentMediaPipeDetectionError,
  MediaPipeHandsRunner,
  MediaPipeHandsRunnerBrokenError,
  MediaPipeHandsRunnerDisposedError,
  MissingMediaPipeResultsError,
  type MediaPipeHandsLike,
} from "./mediaPipeHandsRunner";

interface TestResults {
  id: number;
}

interface TestImage {
  frame: number;
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

class FakeHands implements MediaPipeHandsLike<TestResults, TestImage> {
  listener: ((results: TestResults) => void) | null = null;
  send = vi.fn<(inputs: { image: TestImage }) => Promise<void>>(() =>
    Promise.resolve()
  );
  close = vi.fn<() => Promise<void>>(() => Promise.resolve());

  onResults(listener: (results: TestResults) => void) {
    this.listener = listener;
  }

  emit(results: TestResults) {
    this.listener?.(results);
  }
}

describe("MediaPipeHandsRunner", () => {
  it("pairs a synchronous callback with exactly one send", async () => {
    const hands = new FakeHands();
    hands.send.mockImplementation(async () => {
      hands.emit({ id: 1 });
    });
    const runner = new MediaPipeHandsRunner(hands);

    await expect(runner.detect({ frame: 1 })).resolves.toEqual({ id: 1 });
    expect(hands.send).toHaveBeenCalledTimes(1);
    expect(runner.hasInFlightDetection).toBe(false);
  });

  it("waits for a callback that arrives after send resolves", async () => {
    const hands = new FakeHands();
    const runner = new MediaPipeHandsRunner(hands, { resultTimeoutMs: 100 });

    const detection = runner.detect({ frame: 2 });
    await vi.waitFor(() => expect(hands.send).toHaveBeenCalledTimes(1));
    hands.emit({ id: 2 });

    await expect(detection).resolves.toEqual({ id: 2 });
  });

  it("rejects a concurrent detect without disturbing the active request", async () => {
    const hands = new FakeHands();
    const sendGate = deferred<void>();
    hands.send.mockReturnValue(sendGate.promise);
    const runner = new MediaPipeHandsRunner(hands);

    const first = runner.detect({ frame: 1 });
    await expect(runner.detect({ frame: 2 })).rejects.toBeInstanceOf(
      ConcurrentMediaPipeDetectionError
    );

    hands.emit({ id: 1 });
    sendGate.resolve();
    await expect(first).resolves.toEqual({ id: 1 });
    expect(hands.send).toHaveBeenCalledTimes(1);
  });

  it("uses the first callback and ignores duplicates", async () => {
    const hands = new FakeHands();
    const sendGate = deferred<void>();
    hands.send.mockReturnValue(sendGate.promise);
    const runner = new MediaPipeHandsRunner(hands);

    const detection = runner.detect({ frame: 1 });
    hands.emit({ id: 10 });
    hands.emit({ id: 11 });
    sendGate.resolve();

    await expect(detection).resolves.toEqual({ id: 10 });
  });

  it("marks a send failure as broken and rejects subsequent work", async () => {
    const hands = new FakeHands();
    const sendError = new Error("send failed");
    hands.send.mockRejectedValue(sendError);
    const runner = new MediaPipeHandsRunner(hands);

    await expect(runner.detect({ frame: 1 })).rejects.toBe(sendError);
    expect(runner.isBroken).toBe(true);
    expect(runner.brokenCause).toBe(sendError);
    await expect(runner.detect({ frame: 2 })).rejects.toBeInstanceOf(
      MediaPipeHandsRunnerBrokenError
    );
    expect(hands.send).toHaveBeenCalledTimes(1);
  });

  it("marks a missing callback as broken after the result timeout", async () => {
    vi.useFakeTimers();
    try {
      const hands = new FakeHands();
      const runner = new MediaPipeHandsRunner(hands, { resultTimeoutMs: 25 });
      const detection = runner.detect({ frame: 1 });
      const rejection = expect(detection).rejects.toBeInstanceOf(
        MissingMediaPipeResultsError
      );

      await Promise.resolve();
      await Promise.resolve();
      await vi.advanceTimersByTimeAsync(25);

      await rejection;
      expect(runner.isBroken).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("dispose waits for callback and send before closing", async () => {
    const hands = new FakeHands();
    const sendGate = deferred<void>();
    hands.send.mockReturnValue(sendGate.promise);
    const runner = new MediaPipeHandsRunner(hands);

    const detection = runner.detect({ frame: 1 });
    const disposal = runner.dispose();
    expect(runner.isDisposing).toBe(true);
    expect(hands.close).not.toHaveBeenCalled();

    // Disposal does not discard the transport callback, even though a business
    // session would already have stopped publishing its result.
    hands.emit({ id: 7 });
    await Promise.resolve();
    expect(hands.close).not.toHaveBeenCalled();

    sendGate.resolve();
    await expect(detection).resolves.toEqual({ id: 7 });
    await disposal;

    expect(hands.close).toHaveBeenCalledTimes(1);
    expect(runner.isDisposed).toBe(true);
    await expect(runner.detect({ frame: 2 })).rejects.toBeInstanceOf(
      MediaPipeHandsRunnerDisposedError
    );
  });

  it("dispose is idempotent", async () => {
    const hands = new FakeHands();
    const runner = new MediaPipeHandsRunner(hands);

    const first = runner.dispose();
    const second = runner.dispose();
    expect(second).toBe(first);
    await first;
    expect(hands.close).toHaveBeenCalledTimes(1);
  });
});
