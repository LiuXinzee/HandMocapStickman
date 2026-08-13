import { afterEach, describe, expect, it, vi } from "vitest";
import {
  closeSerialResources,
  runSerialPortTransition,
  waitForPendingSerialClosures,
  type NullableRef,
  type SerialPortResource,
  type SerialReaderResource,
} from "./serialPortLifecycle";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function ref<T>(current: T | null): NullableRef<T> {
  return { current };
}

afterEach(async () => {
  await waitForPendingSerialClosures();
});

describe("closeSerialResources", () => {
  it("detaches refs synchronously and closes resources in strict order", async () => {
    const events: string[] = [];
    const readLoop = deferred<void>();
    const trackedReadLoop = readLoop.promise.then(() => {
      events.push("read-loop");
    });
    const reader: SerialReaderResource = {
      cancel: vi.fn(async () => {
        events.push("cancel");
      }),
      releaseLock: vi.fn(() => {
        events.push("release-lock");
      }),
    };
    const port: SerialPortResource = {
      close: vi.fn(async () => {
        events.push("close-port");
      }),
    };
    const readerRef = ref(reader);
    const portRef = ref(port);
    const readLoopPromiseRef = ref<Promise<unknown>>(trackedReadLoop);

    const cleanup = closeSerialResources(
      readerRef,
      portRef,
      readLoopPromiseRef
    );

    expect(readerRef.current).toBeNull();
    expect(portRef.current).toBeNull();
    expect(readLoopPromiseRef.current).toBeNull();

    await Promise.resolve();
    expect(events).toEqual(["cancel"]);

    readLoop.resolve();
    await cleanup;

    expect(events).toEqual([
      "cancel",
      "read-loop",
      "release-lock",
      "close-port",
    ]);
  });

  it("blocks the next transition until a failed close can be retried", async () => {
    const events: string[] = [];
    let closeAttempts = 0;
    const reader: SerialReaderResource = {
      cancel: vi.fn(async () => {
        events.push("cancel");
        throw new Error("cancel failed");
      }),
      releaseLock: vi.fn(() => {
        events.push("release-lock");
        throw new Error("release failed");
      }),
    };
    const port: SerialPortResource = {
      close: vi.fn(async () => {
        events.push("close-port");
        closeAttempts++;
        if (closeAttempts < 3) throw new Error("close failed");
      }),
    };
    const failedReadLoop = Promise.reject(new Error("read failed"));
    void failedReadLoop.catch(() => undefined);

    const cleanup = closeSerialResources(
      ref(reader),
      ref(port),
      ref<Promise<unknown>>(failedReadLoop)
    );
    await expect(cleanup).rejects.toThrow("close failed");

    let nextTransitionRan = false;
    const nextTransition = () =>
      runSerialPortTransition(async () => {
        nextTransitionRan = true;
      });

    await expect(nextTransition()).rejects.toThrow("close failed");
    expect(nextTransitionRan).toBe(false);
    await expect(nextTransition()).resolves.toBeUndefined();

    expect(events).toEqual([
      "cancel",
      "release-lock",
      "close-port",
      "close-port",
      "close-port",
    ]);
    expect(nextTransitionRan).toBe(true);
  });

  it("continues cleanup when cancel, read, or release fail", async () => {
    const events: string[] = [];
    const reader: SerialReaderResource = {
      cancel: vi.fn(async () => {
        events.push("cancel");
        throw new Error("cancel failed");
      }),
      releaseLock: vi.fn(() => {
        events.push("release-lock");
        throw new Error("release failed");
      }),
    };
    const port: SerialPortResource = {
      close: vi.fn(async () => {
        events.push("close-port");
      }),
    };
    const failedReadLoop = Promise.reject(new Error("read failed"));
    void failedReadLoop.catch(() => undefined);

    await expect(
      closeSerialResources(
        ref(reader),
        ref(port),
        ref<Promise<unknown>>(failedReadLoop)
      )
    ).resolves.toBeUndefined();

    expect(events).toEqual(["cancel", "release-lock", "close-port"]);
  });

  it("does not touch captured resources again when cleanup is repeated", async () => {
    const reader: SerialReaderResource = {
      cancel: vi.fn(async () => undefined),
      releaseLock: vi.fn(),
    };
    const port: SerialPortResource = {
      close: vi.fn(async () => undefined),
    };
    const readerRef = ref(reader);
    const portRef = ref(port);

    const first = closeSerialResources(readerRef, portRef);
    const second = closeSerialResources(readerRef, portRef);
    await Promise.all([first, second]);

    expect(reader.cancel).toHaveBeenCalledTimes(1);
    expect(reader.releaseLock).toHaveBeenCalledTimes(1);
    expect(port.close).toHaveBeenCalledTimes(1);
  });

  it("exposes pending cleanup through the global closure barrier", async () => {
    const readLoop = deferred<void>();
    const cleanup = closeSerialResources(
      ref<SerialReaderResource>(null),
      ref<SerialPortResource>(null),
      ref<Promise<unknown>>(readLoop.promise)
    );
    let barrierFinished = false;
    const barrier = waitForPendingSerialClosures().then(() => {
      barrierFinished = true;
    });

    await Promise.resolve();
    expect(barrierFinished).toBe(false);

    readLoop.resolve();
    await Promise.all([cleanup, barrier]);
    expect(barrierFinished).toBe(true);
  });
});

describe("runSerialPortTransition", () => {
  it("serializes open and close transitions without interleaving", async () => {
    const events: string[] = [];
    const openGate = deferred<void>();
    const reader: SerialReaderResource = {
      cancel: vi.fn(async () => {
        events.push("cancel");
      }),
      releaseLock: vi.fn(() => {
        events.push("release-lock");
      }),
    };
    const port: SerialPortResource = {
      close: vi.fn(async () => {
        events.push("close-port");
      }),
    };

    const firstOpen = runSerialPortTransition(async () => {
      events.push("first-open-start");
      await openGate.promise;
      events.push("first-open-end");
    });
    const cleanup = closeSerialResources(ref(reader), ref(port));
    const nextOpen = runSerialPortTransition(async () => {
      events.push("next-open");
    });

    await Promise.resolve();
    expect(events).toEqual(["first-open-start"]);

    openGate.resolve();
    await Promise.all([firstOpen, cleanup, nextOpen]);

    expect(events).toEqual([
      "first-open-start",
      "first-open-end",
      "cancel",
      "release-lock",
      "close-port",
      "next-open",
    ]);
  });

  it("keeps the queue usable after a failed transition", async () => {
    const first = runSerialPortTransition(async () => {
      throw new Error("open failed");
    });
    const second = runSerialPortTransition(async () => "connected");

    await expect(first).rejects.toThrow("open failed");
    await expect(second).resolves.toBe("connected");
  });
});
