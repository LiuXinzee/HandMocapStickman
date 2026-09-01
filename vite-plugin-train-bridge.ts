/**
 * 句子模型训练桥 —— 把「浏览器导出数据集 → 手动拷进 python_train/data/ → 开终端跑
 * train_seq.py → 结果只在 stdout 里」这一圈缩成网页上的两个按钮。
 *
 * ## 为什么是 Vite 插件，而不是 server/index.ts
 *
 * `server/index.ts` 是**生产**静态服务器。「收文件 + spawn 进程」放进生产产物就是
 * 一个远程代码执行面。挂在 `configureServer` 里，这些路由只在 `npm run dev` 存在，
 * `vite build` 的产物里一行都没有（`apply: "serve"`）。
 *
 * 训练本来就是开发期动作：要本机的 python_train/.venv（TF 只装在那里）、要本机的
 * 数据目录。它不属于部署产物。
 *
 * ## 两道防护，缺一不可
 *
 * `vite.config.ts` 里 `server.host: true` —— dev server 监听 0.0.0.0，同网段任何人
 * 都能打到这些路由。所以每个请求都要过：
 *
 * 1. **来源必须是回环**。
 * 2. **必须带本次启动生成的随机 token**（由 `transformIndexHtml` 注进页面）。
 *
 * 少了这两条，等于给一台对局域网开放的服务加了「上传任意文件 + 起任意进程」。
 *
 * 参数一律走白名单 + 数值校验，`argv` 数组交给 `spawn`（`shell: false`）——
 * 请求体里的字符串**一个都不进命令行**。
 *
 * 形状照 `vite.config.ts` 里已有的 `vitePluginManusDebugCollector`：
 * `server.middlewares.use()` + 手动读 req 流。
 */
import { spawn, type ChildProcess } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin, ViteDevServer } from "vite";

const ROUTE_BASE = "/__train__";
const TOKEN_HEADER = "x-train-token";

/** 日志环形缓冲上限。一次 80 epoch 的训练约 100 行，5000 行够放几十次 */
const MAX_LINES = 5000;
/** 数据集上传上限。实测 dataset.bin 约 30~60 MB，512 MB 是防跑飞不是防大小 */
const MAX_UPLOAD_BYTES = 512 * 1024 * 1024;

/**
 * `export` 与另外两个不同：它跑的是 `export_weights.py` 而不是 `train_seq.py`。
 * 之所以仍然共用同一个 `run` 槽位 —— 训练正在往 `out/sentence_student.keras` 写的
 * 时候去读它，读到的是半个文件。共用槽位天然把两者互斥掉了。
 */
type Target = "ctc" | "distill" | "export";

interface RunState {
  runId: string;
  target: Target;
  /** 实际执行的命令行，原样回显给页面 —— 页面上看到的必须是真跑的那条 */
  argv: string[];
  startedAt: number;
  child: ChildProcess | null;
  lines: string[];
  exitCode: number | null;
  /** 起不来的原因（找不到 python 等）。与 exitCode 分开：一个是没跑，一个是跑完了 */
  error: string | null;
}

interface RunOptions {
  target: Target;
  epochs?: number;
  noSynth?: boolean;
  synthPerTemplate?: number;
  noTrim?: boolean;
}

// ===== 防护 =====

function isLoopback(req: IncomingMessage): boolean {
  const a = req.socket.remoteAddress ?? "";
  return (
    a === "127.0.0.1" ||
    a === "::1" ||
    a === "::ffff:127.0.0.1" ||
    a.startsWith("127.")
  );
}

// ===== 小工具 =====

function sendJson(res: ServerResponse, code: number, body: unknown): void {
  const s = JSON.stringify(body);
  res.writeHead(code, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Length": Buffer.byteLength(s),
  });
  res.end(s);
}

function readBody(req: IncomingMessage, limit: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let n = 0;
    req.on("data", (c: Buffer) => {
      n += c.length;
      if (n > limit) {
        reject(new Error(`请求体超过 ${limit} 字节`));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

/**
 * 找 python_train/.venv 的解释器。
 *
 * **找不到就报错，绝不回落到全局 python** —— 全局那个是共享科研环境、没装 TF，
 * 回落只会得到一句 `ModuleNotFoundError: tensorflow`，而真正的问题是 venv 没建
 * 或者路径不对。回落把一个一眼能看懂的错变成一个要查半天的错。
 */
function findPython(pyDir: string): { path: string } | { error: string } {
  const candidates = [
    path.join(pyDir, ".venv", "Scripts", "python.exe"),
    path.join(pyDir, ".venv", "bin", "python"),
    path.join(pyDir, ".venv", "bin", "python3"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return { path: c };
  }
  return {
    error:
      `找不到 python_train/.venv 的解释器（试过 ${candidates
        .map((c) => path.relative(pyDir, c))
        .join(" / ")}）。` +
      `TensorFlow 只装在这个 venv 里，全局 python 没有，所以不回落。先建 venv。`,
  };
}

/**
 * 由白名单拼 argv。请求体里的字符串一个都不进来 —— 只有布尔开关和过了
 * `Number.isInteger` + 范围校验的数字。
 */
function buildArgv(opts: RunOptions): string[] | { error: string } {
  // 导出走另一个脚本，且**一个参数都不来自请求体** —— 路径全是常量。
  // `--out` 与 export_weights.py 的默认值一致：cwd 是 python_train/，
  // 所以它落到 client/public/models/seq_sentence/，也就是 sentenceModel.ts
  // 读的那个目录（SENTENCE_MODEL_DIR）。两边对不上的话表现是"导出成功但页面说没模型"
  if (opts.target === "export") {
    return [
      "export_weights.py",
      "--model",
      "out/sentence_student.keras",
      "--meta",
      "out/sentence_student_meta.json",
      "--out",
      "../client/public/models/seq_sentence",
    ];
  }

  const argv: string[] = ["train_seq.py", "--data", "data", "--out", "out"];

  if (opts.target === "ctc") {
    argv.push("--ctc", "--events", "out/ctc_events.jsonl");
  } else if (opts.target === "distill") {
    argv.push("--distill");
  } else {
    return { error: `未知的 target` };
  }

  const int = (v: unknown, lo: number, hi: number, flag: string) => {
    if (v === undefined || v === null) return null;
    if (typeof v !== "number" || !Number.isInteger(v) || v < lo || v > hi) {
      return { error: `${flag} 必须是 ${lo}~${hi} 的整数` };
    }
    return String(v);
  };

  const ep = int(opts.epochs, 1, 2000, "--epochs");
  if (ep && typeof ep === "object") return ep;
  if (ep) argv.push("--epochs", ep);

  if (opts.target === "ctc") {
    if (opts.noSynth) argv.push("--no-synth");
    if (opts.noTrim) argv.push("--no-trim");
    const spt = int(opts.synthPerTemplate, 1, 500, "--synth-per-template");
    if (spt && typeof spt === "object") return spt;
    if (spt) argv.push("--synth-per-template", spt);
  }

  return argv;
}

// ===== 插件 =====

export function vitePluginTrainBridge(): Plugin {
  const token = crypto.randomBytes(24).toString("hex");
  let pyDir = "";
  /** `client/public/models/seq_sentence` —— export_weights.py 的落点，也是浏览器读的那个 */
  let modelDir = "";
  let run: RunState | null = null;
  const sseClients = new Set<ServerResponse>();

  function pushLine(line: string): void {
    if (!run) return;
    run.lines.push(line);
    if (run.lines.length > MAX_LINES) run.lines.splice(0, run.lines.length - MAX_LINES);
    fanout("line", line);
  }

  /**
   * 推给所有 SSE 客户端。用 `forEach` 而不是 `for...of`：本仓库的 tsconfig 没设
   * `target`，遍历 Set 会撞上 TS2802（要 downlevelIteration）。为一行循环去动
   * 全项目的 target 不值得
   */
  function fanout(event: string, data: unknown): void {
    const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    sseClients.forEach((c) => c.write(payload));
  }

  /**
   * 把 chunk 流切成行。**必须留住不完整的尾巴** —— 直接对每个 chunk 做 split
   * 会把一行 epoch 日志劈成两条，页面上表现为「偶尔有半行乱码」，而且是随机出现的
   */
  function lineSplitter(emit: (line: string) => void): (c: Buffer) => void {
    let tail = "";
    return (c: Buffer) => {
      tail += c.toString("utf8");
      const parts = tail.split(/\r?\n/);
      tail = parts.pop() ?? "";
      for (const p of parts) emit(p);
    };
  }

  function startRun(opts: RunOptions): { ok: true; state: RunState } | { ok: false; code: number; error: string } {
    if (run?.child) {
      return { ok: false, code: 409, error: "已经有一个训练在跑，先停掉它" };
    }
    const argvOrErr = buildArgv(opts);
    if (!Array.isArray(argvOrErr)) {
      return { ok: false, code: 400, error: argvOrErr.error };
    }
    const py = findPython(pyDir);
    if ("error" in py) {
      return { ok: false, code: 500, error: py.error };
    }

    const state: RunState = {
      runId: crypto.randomBytes(8).toString("hex"),
      target: opts.target,
      argv: [py.path, ...argvOrErr],
      startedAt: Date.now(),
      child: null,
      lines: [],
      exitCode: null,
      error: null,
    };
    run = state;

    const child = spawn(py.path, argvOrErr, {
      cwd: pyDir,
      shell: false, // 绝不过 shell：argv 数组直传，没有任何拼串
      env: {
        ...process.env,
        // stdout 不是 tty 时 Python 全缓冲 —— 页面会十几分钟一片空白，
        // 然后训练结束一次性刷出全部日志。看起来完全像"卡死了"
        PYTHONUNBUFFERED: "1",
        // train_seq.py 的 print 全是中文 + ⚠️。它的 main() 里已经
        // `reconfigure(encoding="utf-8", errors="replace")` 兜住了 —— 但那要等
        // main() 跑起来，在那之前 TF 自己的 import 期输出不受保护。
        // 而且下面按 utf8 解字节的是 node，两端必须约定同一个编码
        PYTHONIOENCODING: "utf-8",
      },
    });
    state.child = child;

    child.stdout?.on("data", lineSplitter(pushLine));
    child.stderr?.on("data", lineSplitter(pushLine));

    child.on("error", (e) => {
      state.error = String(e);
      state.child = null;
      pushLine(`[起进程失败] ${e}`);
      fanout("done", { runId: state.runId, exitCode: null, error: state.error });
    });

    child.on("close", (code) => {
      state.exitCode = code;
      state.child = null;
      fanout("done", { runId: state.runId, exitCode: code, error: null });
    });

    return { ok: true, state };
  }

  return {
    name: "train-bridge",
    // 只在 dev server 生效。构建产物里没有这些路由 —— 这是"收文件 + spawn"
    // 不该出现在部署面的唯一保证
    apply: "serve",

    // 把 token 注进页面。返回**标签数组**而不是 `{ tags }` ——
    // `{ tags }` 那一支要求同时给 `html`，只给 tags 会被类型拒掉
    transformIndexHtml() {
      return [
        {
          tag: "script",
          children: `window.__TRAIN_BRIDGE__=${JSON.stringify({
            token,
            base: ROUTE_BASE,
          })};`,
          injectTo: "head" as const,
        },
      ];
    },

    configureServer(server: ViteDevServer) {
      pyDir = path.resolve(server.config.root, "..", "python_train");
      // config.root 是 client/。publicDir 下的东西 dev server 直接按 / 服务，
      // 所以导出完不需要重启也不需要拷贝
      modelDir = path.resolve(server.config.root, "public", "models", "seq_sentence");

      // 不接 next：这个前缀下的路由全由我们兜底（未知路径回 404），
      // 放过去只会掉到 Vite 的 SPA fallback、拿到一份 index.html
      server.middlewares.use(ROUTE_BASE, (req, res) => {
        // connect 的 use(prefix, fn) 会把前缀从 req.url 上摘掉
        const url = req.url ?? "/";
        const q = url.indexOf("?");
        const route = (q >= 0 ? url.slice(0, q) : url).replace(/\/+$/, "") || "/";
        const method = req.method ?? "GET";

        if (!isLoopback(req)) {
          return sendJson(res, 403, {
            error: "训练桥只接受本机请求。dev server 监听 0.0.0.0，所以这条必须挡。",
          });
        }
        // 头**或**查询串。`EventSource` 带不了自定义头，/stream 只能走查询串 ——
        // 少了这一支，实时日志会稳定 403，而页面上只表现为"日志一直是空的"
        const headerToken = req.headers[TOKEN_HEADER];
        const provided =
          (typeof headerToken === "string" ? headerToken : null) ??
          (q >= 0 ? new URLSearchParams(url.slice(q + 1)).get("token") : null);
        if (provided !== token) {
          return sendJson(res, 403, {
            error: "token 不对。页面是 dev server 启动前打开的？刷新一下。",
          });
        }

        void handle(route, method, req, res).catch((e) => {
          if (!res.headersSent) sendJson(res, 500, { error: String(e) });
        });
        return undefined;
      });

      async function handle(
        route: string,
        method: string,
        req: IncomingMessage,
        res: ServerResponse
      ): Promise<void> {
        // ---- 探活 ----
        if (route === "/ping" || route === "/") {
          const py = findPython(pyDir);
          return sendJson(res, 200, {
            ok: true,
            pythonReady: !("error" in py),
            pythonError: "error" in py ? py.error : null,
            dataDir: path.join(pyDir, "data"),
          });
        }

        // ---- 状态（页面刷新后接回来。SSE 自己不带历史）----
        if (route === "/status") {
          return sendJson(res, 200, {
            running: !!run?.child,
            runId: run?.runId ?? null,
            target: run?.target ?? null,
            argv: run?.argv ?? null,
            startedAt: run?.startedAt ?? null,
            exitCode: run?.exitCode ?? null,
            error: run?.error ?? null,
            lines: run?.lines ?? [],
          });
        }

        // ---- 数据集直送 ----
        if (route === "/dataset" && method === "POST") {
          const body = await readBody(req, MAX_UPLOAD_BYTES);
          if (body.length < 4) return sendJson(res, 400, { error: "请求体太短" });
          const jsonLen = body.readUInt32LE(0);
          if (jsonLen <= 0 || 4 + jsonLen > body.length) {
            return sendJson(res, 400, { error: "manifest 长度头与请求体不符" });
          }
          const manifestText = body.subarray(4, 4 + jsonLen).toString("utf8");
          const bin = body.subarray(4 + jsonLen);
          try {
            JSON.parse(manifestText); // 只验能不能解析，不动内容
          } catch (e) {
            return sendJson(res, 400, { error: `manifest 不是合法 JSON：${e}` });
          }

          const dataDir = path.join(pyDir, "data");
          fs.mkdirSync(dataDir, { recursive: true });
          // 先写 .tmp 再 rename：半个 bin 和一个新 json 并排放着的话，
          // load_dataset 的 _check_pair 会拦住，但报出来的错指向数据不指向上传
          const tmpBin = path.join(dataDir, "dataset.bin.tmp");
          const tmpJson = path.join(dataDir, "dataset.json.tmp");
          fs.writeFileSync(tmpBin, bin);
          fs.writeFileSync(tmpJson, manifestText, "utf8");
          fs.renameSync(tmpBin, path.join(dataDir, "dataset.bin"));
          fs.renameSync(tmpJson, path.join(dataDir, "dataset.json"));

          return sendJson(res, 200, {
            ok: true,
            binBytes: bin.length,
            jsonBytes: Buffer.byteLength(manifestText),
            dataDir,
          });
        }

        // ---- 起训练 ----
        if (route === "/run" && method === "POST") {
          const raw = await readBody(req, 64 * 1024);
          let opts: RunOptions;
          try {
            opts = JSON.parse(raw.toString("utf8") || "{}") as RunOptions;
          } catch (e) {
            return sendJson(res, 400, { error: `请求体不是合法 JSON：${e}` });
          }
          const r = startRun(opts);
          if (!r.ok) return sendJson(res, r.code, { error: r.error });
          return sendJson(res, 200, {
            runId: r.state.runId,
            // 回显真实命令行：页面上显示的必须是真跑的那条，不是页面自己拼的
            argv: r.state.argv,
          });
        }

        // ---- 停 ----
        if (route === "/stop" && method === "POST") {
          if (!run?.child) return sendJson(res, 200, { ok: true, wasRunning: false });
          const child = run.child;
          child.kill("SIGTERM");
          // Windows 上 node 的 kill 走 TerminateProcess，SIGTERM 已经是硬杀；
          // POSIX 上给 3 秒收尾再补一刀
          const timer = setTimeout(() => {
            if (!child.killed) child.kill("SIGKILL");
          }, 3000);
          child.once("close", () => clearTimeout(timer));
          pushLine("[已请求停止]");
          return sendJson(res, 200, { ok: true, wasRunning: true });
        }

        // ---- 日志流 ----
        if (route === "/stream") {
          res.writeHead(200, {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            Connection: "keep-alive",
            // dev server 前面可能有代理，这一条防它攒缓冲
            "X-Accel-Buffering": "no",
          });
          res.write(": connected\n\n");
          sseClients.add(res);
          // 心跳：没有它的话空闲连接会被中间层静默掐掉，
          // 而症状是"训练跑到一半日志就不动了"
          const beat = setInterval(() => res.write(": beat\n\n"), 15000);
          req.on("close", () => {
            clearInterval(beat);
            sseClients.delete(res);
          });
          return;
        }

        // ---- 结果 meta ----
        if (route === "/result") {
          const p = path.join(pyDir, "out", "sentence_student_meta.json");
          if (!fs.existsSync(p)) {
            return sendJson(res, 404, { error: "还没有 sentence_student_meta.json" });
          }
          const text = fs.readFileSync(p, "utf8");
          res.writeHead(200, {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
          });
          res.end(text);
          return;
        }

        // ---- 导出权重到浏览器能读的目录 ----
        if (route === "/export" && method === "POST") {
          // 先自己查一遍。少了这一条会是：spawn 一个 python、等它 import 完 TF
          // （二十来秒）、才在最后打出「找不到 out/sentence_student.keras」。
          // 而这二十秒里页面显示的是"正在导出"
          const keras = path.join(pyDir, "out", "sentence_student.keras");
          if (!fs.existsSync(keras)) {
            return sendJson(res, 400, {
              error: "还没有 out/sentence_student.keras —— 先训一次 CTC 模型再导出。",
            });
          }
          const r = startRun({ target: "export" });
          if (!r.ok) return sendJson(res, r.code, { error: r.error });
          return sendJson(res, 200, { runId: r.state.runId, argv: r.state.argv });
        }

        // ---- 浏览器端当前部署的是哪一份 ----
        if (route === "/model") {
          const wj = path.join(modelDir, "weights.json");
          const wb = path.join(modelDir, "weights.bin");
          if (!fs.existsSync(wj) || !fs.existsSync(wb)) {
            return sendJson(res, 200, { exists: false, modelDir });
          }
          let meta: unknown = null;
          try {
            meta = (JSON.parse(fs.readFileSync(wj, "utf8")) as { meta?: unknown }).meta ?? null;
          } catch {
            /* 文件坏了也要能回一个 exists:true —— 页面据此提示"重新导出" */
          }
          // trainedAt 是 out/ 那份 keras 的时间。它比 deployedAt 新就说明训完还没导，
          // 这正是这条路由存在的理由：页面上那些指标来自 out/，而 /translate 用的是
          // 这个目录 —— 两者不是同一份模型，看起来却一模一样
          const keras = path.join(pyDir, "out", "sentence_student.keras");
          return sendJson(res, 200, {
            exists: true,
            modelDir,
            deployedAt: fs.statSync(wj).mtimeMs,
            bytes: fs.statSync(wb).size,
            trainedAt: fs.existsSync(keras) ? fs.statSync(keras).mtimeMs : null,
            meta,
          });
        }

        /*
         * ---- 词骨干的状态 ----
         *
         * CTC 训练一开头就 `transfer_backbone(model, out/student.keras)`，而那一步
         * **决定收不收敛**（CTC 的梯度比分类稀疏得多，随机初始化起跑几百条句子训不动）。
         * 它的两种失败都是静默的：
         *   文件不存在 → 只打一行 ⚠️ 然后照跑，表现是 loss 半天不降
         *   文件旧了   → 照样打印「骨干迁移 N 层」，因为 conv/bn 的形状**不随类别数变**，
         *                 骨干没见过新词也照搬成功。表现是那个词的 WER 特别差
         * 后一种是这条路由存在的理由。
         *
         * **只回原始事实，不判断。** `missing` 的计算要减掉 `UNTRAINED_WORDS`
         * （那份清单在 client/src/lib/sentenceTemplates.ts，由测试锁着），
         * 放在浏览器侧算 —— 桥去 import client 的 TS 会变成第二份清单，
         * 而两份清单不一致的表现是页面上永远挂着一个假的「骨干缺 happy」。
         *
         * 两个 labels 都是**原始标签**、可直接比：
         *   student_meta.json.labels 是词模型的类别表（`label_to_idx` 直接用 labels，
         *     不过 merge_label —— 所以里面 he/you/we 是分开的）
         *   dataset.json.labels 是导出序列的 primaryLabel 去重（datasetStore 那边）
         * CTC 自己用的是 merged_class_table，但那只影响 `*_out` 那一层，
         * 而 transfer_backbone 正是跳过它的。
         */
        if (route === "/backbone") {
          const keras = path.join(pyDir, "out", "student.keras");
          const metaPath = path.join(pyDir, "out", "student_meta.json");
          const dsPath = path.join(pyDir, "data", "dataset.json");

          let backboneLabels: string[] | null = null;
          let seqLen: number | null = null;
          try {
            const m = JSON.parse(fs.readFileSync(metaPath, "utf8")) as {
              labels?: unknown;
              seqLen?: unknown;
            };
            if (Array.isArray(m.labels)) backboneLabels = m.labels.map(String);
            if (typeof m.seqLen === "number") seqLen = m.seqLen;
          } catch {
            /* 没有 meta 或坏了 —— 下面 backboneLabels 保持 null，页面据此显示"查不到" */
          }

          let datasetLabels: string[] | null = null;
          try {
            const d = JSON.parse(fs.readFileSync(dsPath, "utf8")) as { labels?: unknown };
            if (Array.isArray(d.labels)) datasetLabels = d.labels.map(String);
          } catch {
            /* 还没直送过数据集 */
          }

          return sendJson(res, 200, {
            exists: fs.existsSync(keras),
            // .keras 与 _meta.json 是同一次训练写的两个文件，取 keras 的时间：
            // 页面要比的是"骨干训于何时"，而 meta 可能被手工改过
            trainedAt: fs.existsSync(keras) ? fs.statSync(keras).mtimeMs : null,
            datasetAt: fs.existsSync(dsPath) ? fs.statSync(dsPath).mtimeMs : null,
            backboneLabels,
            datasetLabels,
            seqLen,
          });
        }

        // ---- 结构化事件 ----
        if (route === "/events") {
          const p = path.join(pyDir, "out", "ctc_events.jsonl");
          if (!fs.existsSync(p)) return sendJson(res, 200, { events: [] });
          const text = fs.readFileSync(p, "utf8");
          res.writeHead(200, {
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-store",
          });
          res.end(text);
          return;
        }

        return sendJson(res, 404, { error: `未知路由 ${route}` });
      }

      // dev server 关掉时把子进程带走，否则一个 TF 进程会挂在后台占着 GPU
      server.httpServer?.on("close", () => {
        run?.child?.kill("SIGKILL");
      });
    },
  };
}
