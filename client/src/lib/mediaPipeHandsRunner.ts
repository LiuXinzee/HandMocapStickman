export interface MediaPipeHandsLike<TResult, TImage = CanvasImageSource> {
  onResults(listener: (results: TResult) => void): void;
  send(inputs: { image: TImage }): Promise<void>;
  close(): Promise<void>;
}

export interface MediaPipeHandsRunnerOptions {
  /**
   * Maximum time to wait for onResults after send() has resolved. MediaPipe
   * normally invokes the listener before send() resolves; this timeout keeps a
   * broken or incompatible implementation from leaving detect() pending.
   */
  resultTimeoutMs?: number;
}

export class ConcurrentMediaPipeDetectionError extends Error {
  constructor() {
    super("A MediaPipe Hands detection is already in progress");
    this.name = "ConcurrentMediaPipeDetectionError";
  }
}

export class MediaPipeHandsRunnerBrokenError extends Error {
  constructor(cause?: unknown) {
    super("The MediaPipe Hands runner is broken and must be disposed");
    this.name = "MediaPipeHandsRunnerBrokenError";
    if (cause !== undefined) this.cause = cause;
  }
}

export class MediaPipeHandsRunnerDisposedError extends Error {
  constructor() {
    super("The MediaPipe Hands runner is disposing or has been disposed");
    this.name = "MediaPipeHandsRunnerDisposedError";
  }
}

export class MissingMediaPipeResultsError extends Error {
  constructor(timeoutMs: number) {
    super(
      `MediaPipe Hands produced no results within ${timeoutMs}ms of send()`
    );
    this.name = "MissingMediaPipeResultsError";
  }
}

interface PendingResult<TResult> {
  callbackSeen: boolean;
  resolve: (results: TResult) => void;
}

const DEFAULT_RESULT_TIMEOUT_MS = 1_000;

/**
 * Serial adapter for MediaPipe Hands' callback-based result API.
 *
 * A runner deliberately permits only one detect() at a time because onResults
 * carries no request identifier. Any send/result failure poisons the runner so
 * a late callback can never be paired with a subsequent request.
 */
export class MediaPipeHandsRunner<TResult, TImage = CanvasImageSource> {
  private readonly resultTimeoutMs: number;
  private pending: PendingResult<TResult> | null = null;
  private inFlight: Promise<TResult> | null = null;
  private disposePromise: Promise<void> | null = null;
  private disposeStarted = false;
  private disposed = false;
  private broken = false;
  private brokenCauseInternal: unknown = null;

  constructor(
    private readonly hands: MediaPipeHandsLike<TResult, TImage>,
    options: MediaPipeHandsRunnerOptions = {}
  ) {
    const timeout = options.resultTimeoutMs ?? DEFAULT_RESULT_TIMEOUT_MS;
    if (!Number.isFinite(timeout) || timeout <= 0) {
      throw new RangeError("resultTimeoutMs must be a positive finite number");
    }
    this.resultTimeoutMs = timeout;

    hands.onResults(results => {
      const pending = this.pending;
      if (!pending || pending.callbackSeen) return;
      pending.callbackSeen = true;
      pending.resolve(results);
    });
  }

  get isBroken(): boolean {
    return this.broken;
  }

  get brokenCause(): unknown {
    return this.brokenCauseInternal;
  }

  get isDisposing(): boolean {
    return this.disposeStarted && !this.disposed;
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  get hasInFlightDetection(): boolean {
    return this.inFlight !== null;
  }

  detect(image: TImage): Promise<TResult> {
    if (this.disposeStarted) {
      return Promise.reject(new MediaPipeHandsRunnerDisposedError());
    }
    if (this.isBroken) {
      return Promise.reject(
        new MediaPipeHandsRunnerBrokenError(this.brokenCauseInternal)
      );
    }
    if (this.inFlight) {
      return Promise.reject(new ConcurrentMediaPipeDetectionError());
    }

    let resolveResult!: (results: TResult) => void;
    const resultPromise = new Promise<TResult>(resolve => {
      resolveResult = resolve;
    });
    const pending: PendingResult<TResult> = {
      callbackSeen: false,
      resolve: resolveResult,
    };
    this.pending = pending;

    // Starting send() in a microtask also converts a synchronous throw from a
    // test double or incompatible implementation into a rejected Promise.
    const sendPromise = Promise.resolve().then(() =>
      this.hands.send({ image })
    );
    const pairedResult = Promise.all([sendPromise, resultPromise]).then(
      ([, results]) => results
    );

    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const missingResult = new Promise<never>((_, reject) => {
      void sendPromise.then(
        () => {
          if (pending.callbackSeen) return;
          timeoutId = setTimeout(() => {
            reject(new MissingMediaPipeResultsError(this.resultTimeoutMs));
          }, this.resultTimeoutMs);
        },
        () => {
          // pairedResult propagates the send failure.
        }
      );
    });

    const operation = Promise.race([pairedResult, missingResult])
      .catch(error => {
        this.broken = true;
        this.brokenCauseInternal = error;
        throw error;
      })
      .finally(() => {
        if (timeoutId !== null) clearTimeout(timeoutId);
        if (this.pending === pending) this.pending = null;
        if (this.inFlight === operation) this.inFlight = null;
      });

    this.inFlight = operation;
    return operation;
  }

  /**
   * Stops accepting work immediately, then lets the current send/result pair
   * settle before closing the underlying MediaPipe graph.
   */
  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposeStarted = true;
    const activeDetection = this.inFlight;

    this.disposePromise = (async () => {
      if (activeDetection) {
        try {
          await activeDetection;
        } catch {
          // The runner is still closed after a failed in-flight detection.
        }
      }
      await this.hands.close();
    })().finally(() => {
      this.disposed = true;
    });

    return this.disposePromise;
  }
}
