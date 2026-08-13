export interface NullableRef<T> {
  current: T | null;
}

export interface SerialReaderResource {
  cancel(): Promise<void>;
  releaseLock(): void;
}

export interface SerialPortResource {
  close(): Promise<void>;
}

let serialPortTransitionTail: Promise<void> = Promise.resolve();
const pendingSerialClosures = new Set<Promise<void>>();
const failedSerialClosures = new Set<SerialPortResource>();

async function retryFailedSerialClosures(): Promise<void> {
  let firstError: unknown = null;

  for (const port of Array.from(failedSerialClosures)) {
    try {
      await port.close();
      failedSerialClosures.delete(port);
    } catch (error) {
      if (firstError === null) firstError = error;
    }
  }

  if (firstError !== null) throw firstError;
}

export function runSerialPortTransition<T>(
  operation: () => Promise<T>
): Promise<T> {
  const result = serialPortTransitionTail.then(() => {
    if (failedSerialClosures.size === 0) return operation();
    return retryFailedSerialClosures().then(operation);
  });
  serialPortTransitionTail = result.then(
    () => undefined,
    () => undefined
  );
  return result;
}

export function closeSerialResources<
  TReader extends SerialReaderResource,
  TPort extends SerialPortResource,
>(
  readerRef: NullableRef<TReader>,
  portRef: NullableRef<TPort>,
  readLoopPromiseRef?: NullableRef<Promise<unknown>>
): Promise<void> {
  const reader = readerRef.current;
  const port = portRef.current;
  const readLoopPromise = readLoopPromiseRef?.current ?? null;

  readerRef.current = null;
  portRef.current = null;
  if (readLoopPromiseRef) readLoopPromiseRef.current = null;

  const cleanup = runSerialPortTransition(async () => {
    try {
      await reader?.cancel();
    } catch {
      // Continue releasing the remaining resources.
    }

    try {
      await readLoopPromise;
    } catch {
      // A read error must not leave the reader lock or port open.
    }

    try {
      reader?.releaseLock();
    } catch {
      // Continue closing the port even if the lock was already released.
    }

    try {
      await port?.close();
    } catch (error) {
      if (port) failedSerialClosures.add(port);
      throw error;
    }
  });

  pendingSerialClosures.add(cleanup);
  void cleanup.then(
    () => pendingSerialClosures.delete(cleanup),
    () => pendingSerialClosures.delete(cleanup)
  );

  return cleanup;
}

export async function waitForPendingSerialClosures(): Promise<void> {
  while (pendingSerialClosures.size > 0) {
    await Promise.all(pendingSerialClosures);
  }
}
