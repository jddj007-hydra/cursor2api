/**
 * One Cursor turn over Connect-RPC / HTTP2.
 *
 * `agent.v1.AgentService/Run` is a bidirectional stream, not a unary call. The
 * exchange for a single turn is:
 *
 *   client -> AgentClientMessage{run_request}
 *   server -> kv_server_message{get_blob_args}            (pulls our history blobs)
 *   client -> kv_client_message{get_blob_result}
 *   server -> exec_server_message{request_context_args}   (tool / rules handshake)
 *   client -> exec_client_message{request_context_result}
 *   server -> interaction_update{text_delta}...           (model output)
 *   server -> exec_server_message{mcp_args}               (model wants a caller tool)
 *   server -> interaction_update{turn_ended}
 *
 * Because the OpenAI protocol executes tools on the *caller* side, we do not
 * answer `mcp_args` here: it is surfaced as a `tool_call` event, the turn ends
 * with finish_reason=tool_calls, and the caller sends the result back on the next
 * request where it is replayed as conversation history.
 */
import { createHash, randomUUID } from "node:crypto";
import * as zlib from "node:zlib";
import type { Config } from "../config.js";
import { connectH2 } from "./h2.js";
import { AgentClientMessage, AgentServerMessage, decodeArgValue, toBytes } from "./proto.js";

export interface McpToolDefinition {
  name: string;
  toolName: string;
  providerIdentifier: string;
  description: string;
  inputSchemaJson: string;
}

export type TurnEvent =
  | { type: "text"; delta: string }
  | { type: "thinking"; delta: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "usage"; inputTokens?: number; outputTokens?: number }
  | { type: "done"; reason: "stop" | "tool_calls" | "error"; error?: string; code?: string };

export interface TurnImage {
  mimeType: string;
  data: Buffer;
  path: string;
  width?: number;
  height?: number;
}

export interface RunTurnOptions {
  token: string;
  model: string;
  /** History entries (oldest first, excluding the active user message). */
  rootMessages: unknown[];
  /** The active user message text. */
  userText: string;
  /** Inline images attached to the active user message. */
  images?: TurnImage[];
  tools: McpToolDefinition[];
  config: Config;
  signal?: AbortSignal;
}

/** Cursor UserMessage, including selected_images when the caller attached any. */
export function buildUserMessage(text: string, images: TurnImage[] | undefined, messageId: string): Record<string, unknown> {
  const userMessage: Record<string, unknown> = { text, messageId };
  if (!images || images.length === 0) return userMessage;
  userMessage.selectedContext = {
    selectedImages: images.map((img) => {
      const selected: Record<string, unknown> = {
        uuid: randomUUID(),
        path: img.path,
        mimeType: img.mimeType,
        data: img.data,
      };
      if (img.width && img.height) selected.dimension = { width: img.width, height: img.height };
      return selected;
    }),
  };
  return userMessage;
}

/**
 * Upstream tool-call ids are not always safe to hand back to a caller: grok models
 * return two ids joined by a literal newline (`call-<uuid>-0\nfc_<uuid>_0`). The
 * caller round-trips this string as `tool_call_id`, and anything that re-encodes it
 * under a stricter schema — Anthropic's `tool_use_id` pattern, for one — rejects the
 * raw form, so fold everything outside the id alphabet into `_`.
 */
function safeToolCallId(raw: unknown): string {
  const id = String(raw ?? "").replace(/[^A-Za-z0-9_-]/g, "_");
  return id || `call_${randomUUID()}`;
}

/** Map an upstream HTTP status onto the Connect code the trailer would have carried. */
function connectCodeForHttp(status: number): string | undefined {
  if (status === 429) return "resource_exhausted";
  if (status === 401) return "unauthenticated";
  if (status === 403) return "permission_denied";
  return undefined;
}

/** providerIdentifier stamped on caller tools; also the name of the one MCP "server" we report. */
export const MCP_PROVIDER = "openai";

/** ExecServerMessage fields that are not part of its `message` oneof. */
const EXEC_ENVELOPE_FIELDS = new Set(["id", "execId", "spanContext", "acceptHookAdditionalContexts"]);

const FLAG_COMPRESSED = 0x01;
const FLAG_END_STREAM = 0x02;

function connectFrame(payload: Uint8Array): Buffer {
  const head = Buffer.alloc(5);
  head[0] = 0;
  head.writeUInt32BE(payload.length, 1);
  return Buffer.concat([head, Buffer.from(payload)]);
}

/** Blob ids are the sha256 of the blob contents. */
function blobId(data: Buffer): Buffer {
  return createHash("sha256").update(data).digest();
}

export async function* runTurn(opts: RunTurnOptions): AsyncGenerator<TurnEvent> {
  const { config } = opts;

  // Every history entry travels as a KV blob that the server pulls back by id.
  const blobs = new Map<string, Buffer>();
  const store = (data: Buffer): Buffer => {
    const id = blobId(data);
    blobs.set(id.toString("hex"), data);
    return id;
  };
  const rootPromptMessagesJson = opts.rootMessages.map((entry) =>
    store(Buffer.from(JSON.stringify(entry), "utf8")),
  );

  const runRequest: Record<string, unknown> = {
    conversationState: {
      rootPromptMessagesJson,
      turns: [],
      todos: [],
      pendingToolCalls: [],
      previousWorkspaceUris: [],
      summaryArchives: [],
      turnTimings: [],
    },
    // A turn driven by a fresh user message uses `userMessageAction`. When the
    // caller is mid-agent-loop (its last message is a tool result, so there is no
    // new user text) the server expects `resumeAction` instead — sending an empty
    // user message there makes the model answer the void.
    action:
      opts.userText.trim() || (opts.images?.length ?? 0) > 0
        ? { userMessageAction: { userMessage: buildUserMessage(opts.userText, opts.images, randomUUID()) } }
        : { resumeAction: {} },
    modelDetails: { modelId: opts.model, displayModelId: opts.model, displayName: opts.model },
    requestedModel: { modelId: opts.model },
    conversationId: randomUUID(),
  };
  if (opts.tools.length > 0) runRequest.mcpTools = { mcpTools: opts.tools };

  const headers: Record<string, string> = {
    ":method": "POST",
    ":path": "/agent.v1.AgentService/Run",
    "content-type": "application/connect+proto",
    "connect-protocol-version": "1",
    te: "trailers",
    authorization: `Bearer ${opts.token}`,
    "x-ghost-mode": "true",
    "x-cursor-client-version": config.clientVersion,
    "x-cursor-client-type": "cli",
    "x-request-id": randomUUID(),
  };
  // The only lever that removes Cursor's native tool registry from the model's view.
  if (config.allowedNativeTools.length > 0) {
    headers["x-cursor-agent-allowed-tools"] = config.allowedNativeTools.join(",");
  }

  const client = await connectH2(config.cursorBaseUrl, config.proxyUrl);
  const req = client.request(headers);

  const declaredTools = new Set(opts.tools.map((t) => t.name));
  let reportedOutputTokens = 0;

  // http2 callbacks feed this queue; the generator body drains it.
  const queue: TurnEvent[] = [];
  let notify: (() => void) | null = null;
  let ended = false;
  const push = (ev: TurnEvent): void => {
    queue.push(ev);
    const n = notify;
    notify = null;
    n?.();
  };
  const end = (ev: TurnEvent): void => {
    if (ended) return;
    ended = true;
    push(ev);
  };

  const send = (message: Record<string, unknown>): void => {
    if (req.closed || req.destroyed) return;
    req.write(connectFrame(AgentClientMessage.encode(AgentClientMessage.create(message)).finish()));
  };
  const sendExec = (ex: Record<string, any>, result: Record<string, unknown>): void =>
    send({
      execClientMessage: {
        id: ex.id,
        ...(ex.execId !== undefined ? { execId: ex.execId } : {}),
        ...result,
      },
    });

  const decode = (payload: Buffer): Record<string, any> =>
    AgentServerMessage.toObject(AgentServerMessage.decode(payload), {
      defaults: false,
      longs: Number,
      enums: String,
      bytes: String, // base64 — normalised through toBytes()
    });

  req.on("response", (h) => {
    const status = Number(h[":status"] || 0);
    if (status !== 200) {
      end({
        type: "done",
        reason: "error",
        code: connectCodeForHttp(status),
        error: `upstream HTTP ${status}`,
      });
    }
  });

  let buffer = Buffer.alloc(0);
  req.on("data", (chunk: Buffer) => {
    buffer = Buffer.concat([buffer, chunk]);
    while (buffer.length >= 5) {
      const flag = buffer[0]!;
      const len = buffer.readUInt32BE(1);
      if (buffer.length < 5 + len) break;
      let payload = buffer.subarray(5, 5 + len);
      buffer = buffer.subarray(5 + len);
      if (flag & FLAG_COMPRESSED) {
        try {
          payload = zlib.gunzipSync(payload);
        } catch {
          /* fall through; decode will fail and we skip the frame */
        }
      }
      if (flag & FLAG_END_STREAM) {
        end(endStreamEvent(payload));
        return;
      }
      let msg: Record<string, any>;
      try {
        msg = decode(payload);
      } catch {
        continue; // unknown frame shape — ignore rather than kill the turn
      }
      handle(msg);
    }
  });

  function endStreamEvent(payload: Buffer): TurnEvent {
    const trailer = payload.toString("utf8");
    try {
      const parsed = JSON.parse(trailer) as { error?: { code?: string; message?: string } };
      if (parsed.error) {
        return {
          type: "done",
          reason: "error",
          code: parsed.error.code,
          error: `${parsed.error.code ?? "error"}: ${parsed.error.message ?? trailer}`,
        };
      }
    } catch {
      if (trailer.trim()) return { type: "done", reason: "error", error: trailer.slice(0, 500) };
    }
    return { type: "done", reason: "stop" };
  }

  function handle(msg: Record<string, any>): void {
    if (msg.kvServerMessage) {
      const kv = msg.kvServerMessage;
      if (kv.getBlobArgs) {
        const id = toBytes(kv.getBlobArgs.blobId).toString("hex");
        const data = blobs.get(id);
        send({ kvClientMessage: { id: kv.id, getBlobResult: data ? { blobData: data } : {} } });
      } else if (kv.setBlobArgs) {
        blobs.set(toBytes(kv.setBlobArgs.blobId).toString("hex"), toBytes(kv.setBlobArgs.blobData));
        send({ kvClientMessage: { id: kv.id, setBlobResult: {} } });
      }
      return;
    }

    if (msg.execServerMessage) {
      const ex = msg.execServerMessage;
      if (ex.requestContextArgs) {
        // Deliberately minimal: no rules, no repo info, no project layout. The only
        // thing advertised is the caller's tool set.
        sendExec(ex, {
          requestContextResult: {
            success: {
              requestContext: {
                rules: [],
                tools: opts.tools,
                env: {
                  osVersion: "linux",
                  workspacePaths: [],
                  shell: "bash",
                  sandboxEnabled: false,
                  timeZone: "UTC",
                },
                repositoryInfo: [],
                gitRepos: [],
                projectLayouts: [],
                mcpInstructions: [],
              },
            },
          },
        });
        return;
      }
      if (ex.mcpStateExecArgs) {
        // The server's GET_MCP_TOOLS tool reads the tool list back from the client.
        // The caller's declared tools are the only "server" we have.
        sendExec(ex, {
          mcpStateExecResult: {
            success: {
              servers: [
                {
                  serverName: MCP_PROVIDER,
                  serverIdentifier: MCP_PROVIDER,
                  tools: opts.tools,
                  instructions: [],
                  status: "connected",
                },
              ],
            },
          },
        });
        return;
      }
      if (ex.mcpAllowlistPrecheckArgs) {
        sendExec(ex, { mcpAllowlistPrecheckResult: { allowlisted: true } });
        return;
      }
      if (ex.shellAllowlistPrecheckArgs) {
        sendExec(ex, { shellAllowlistPrecheckResult: { allowlisted: true } });
        return;
      }
      const native = nativeToolCallFromExec(ex);
      if (native) {
        push({
          type: "tool_call",
          id: safeToolCallId(native.id),
          name: declaredName(native.name, native.aliases, declaredTools),
          args: native.args,
        });
        // Same as mcp_args: the caller runs the tool and replays the result.
        end({ type: "done", reason: "tool_calls" });
        return;
      }
      if (ex.mcpArgs) {
        const a = ex.mcpArgs;
        const args: Record<string, unknown> = {};
        for (const [key, raw] of Object.entries(a.args ?? {})) args[key] = decodeArgValue(raw);
        push({
          type: "tool_call",
          id: safeToolCallId(a.toolCallId),
          name: callerToolName(a, declaredTools),
          args,
        });
        // Hand control back to the caller — it owns tool execution.
        end({ type: "done", reason: "tool_calls" });
        return;
      }
      // Any other exec request is a native Cursor tool the caller cannot run. The
      // server would wait for a result until the deadline, so fail the turn now.
      const kind = Object.keys(ex).find((k) => !EXEC_ENVELOPE_FIELDS.has(k)) ?? "unknown";
      end({
        type: "done",
        reason: "error",
        code: "unimplemented",
        error: `model requested native Cursor tool "${kind}", which this gateway cannot execute; check CURSOR_ALLOWED_NATIVE_TOOLS`,
      });
      return;
    }

    if (msg.interactionQuery) {
      const kind = Object.keys(msg.interactionQuery).find((k) => k !== "id") ?? "unknown";
      end({
        type: "done",
        reason: "error",
        code: "unimplemented",
        error: `upstream sent interaction query "${kind}", which this gateway cannot answer`,
      });
      return;
    }

    if (msg.interactionUpdate) {
      const u = msg.interactionUpdate;
      if (u.textDelta?.text) push({ type: "text", delta: String(u.textDelta.text) });
      else if (u.thinkingDelta?.text) push({ type: "thinking", delta: String(u.thinkingDelta.text) });
      else if (u.tokenDelta) {
        // Each update carries the tokens generated since the previous one.
        const n = finiteOrUndefined(u.tokenDelta.tokens);
        if (n !== undefined) {
          reportedOutputTokens += n;
          push({ type: "usage", outputTokens: reportedOutputTokens });
        }
      } else if (u.turnEnded) end({ type: "done", reason: "stop" });
    }
  }

  req.on("error", (err) => end({ type: "done", reason: "error", error: err.message }));
  req.on("end", () => end({ type: "done", reason: "stop" }));

  const timer = setTimeout(
    () => end({ type: "done", reason: "error", code: "deadline_exceeded", error: "upstream timeout" }),
    config.requestTimeoutMs,
  );
  const onAbort = (): void => end({ type: "done", reason: "error", error: "client aborted" });
  opts.signal?.addEventListener("abort", onAbort, { once: true });

  send({ runRequest });

  // Cursor reports no prompt token count, so estimate what the caller sent. Cursor's
  // own agent prompt is not visible to us and is not included.
  yield {
    type: "usage",
    inputTokens: estimateTokens(
      JSON.stringify(opts.rootMessages) +
        opts.userText +
        JSON.stringify(opts.tools) +
        (opts.images ?? []).map((img) => `${img.mimeType}:${img.data.length}`).join(","),
    ),
  };
  let outputText = "";

  try {
    for (;;) {
      if (queue.length === 0) {
        await new Promise<void>((resolve) => {
          notify = resolve;
        });
      }
      const ev = queue.shift()!;
      if (ev.type === "text" && ev.delta === "") continue;
      if (ev.type === "text" || ev.type === "thinking") outputText += ev.delta;
      else if (ev.type === "tool_call") outputText += ev.name + JSON.stringify(ev.args);
      else if (ev.type === "done" && reportedOutputTokens === 0 && outputText) {
        yield { type: "usage", outputTokens: estimateTokens(outputText) };
      }
      yield ev;
      if (ev.type === "done") return;
    }
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onAbort);
    try {
      req.close();
    } catch {
      /* already closed */
    }
    try {
      client.close();
    } catch {
      /* already closed */
    }
  }
}

/**
 * The server namespaces caller tools as `<providerIdentifier>-<name>` in `name`
 * (e.g. `openai-bash`); the caller only knows the bare name it declared.
 */
/**
 * Cursor natives that correspond to a caller's tool. Grok calls these after a
 * tool result even when the same capability was already exposed as MCP. Bridging
 * them keeps the agent loop on the caller instead of dying with "Tool not available".
 */
export function nativeToolCallFromExec(
  ex: Record<string, any>,
): { name: string; aliases: string[]; args: Record<string, unknown>; id: unknown } | undefined {
  const idOf = (a: Record<string, any>): unknown => a.toolCallId ?? ex.execId ?? ex.id;
  if (ex.piBashArgs || ex.shellArgs || ex.shellStreamArgs || ex.miniSweAgentBashArgs) {
    const a = ex.piBashArgs ?? ex.shellArgs ?? ex.shellStreamArgs ?? ex.miniSweAgentBashArgs;
    // Only Pi's bash timeout is in seconds. ShellArgs.timeout has no documented unit.
    const timeout = ex.piBashArgs ? a.timeout : undefined;
    return { name: "bash", aliases: ["shell"], args: defined({ command: a.command, timeout }), id: idOf(a) };
  }
  if (ex.piReadArgs || ex.readArgs || ex.redactedReadArgs) {
    const a = ex.piReadArgs ?? ex.readArgs ?? ex.redactedReadArgs;
    return { name: "read", aliases: [], args: defined({ path: a.path, offset: a.offset, limit: a.limit }), id: idOf(a) };
  }
  if (ex.piEditArgs) {
    const a = ex.piEditArgs;
    const edits = Array.isArray(a.edits)
      ? a.edits.map((edit: Record<string, any>) => defined({ oldText: edit.oldText, newText: edit.newText }))
      : [];
    return { name: "edit", aliases: [], args: defined({ path: a.path, edits }), id: idOf(a) };
  }
  if (ex.piWriteArgs || ex.writeArgs) {
    const a = ex.piWriteArgs ?? ex.writeArgs;
    return {
      name: "write",
      aliases: [],
      args: defined({ path: a.path, content: a.content ?? a.fileText }),
      id: idOf(a),
    };
  }
  if (ex.piGrepArgs || ex.grepArgs) {
    const a = ex.piGrepArgs ?? ex.grepArgs;
    return {
      name: "grep",
      aliases: [],
      args: defined({
        pattern: a.pattern,
        path: a.path,
        glob: a.glob,
        ignoreCase: a.ignoreCase ?? a.caseInsensitive,
        literal: a.literal,
        context: a.context,
        limit: a.limit ?? a.headLimit,
      }),
      id: idOf(a),
    };
  }
  if (ex.piFindArgs) {
    const a = ex.piFindArgs;
    return { name: "find", aliases: [], args: defined({ pattern: a.pattern, path: a.path, limit: a.limit }), id: idOf(a) };
  }
  if (ex.piLsArgs || ex.lsArgs) {
    const a = ex.piLsArgs ?? ex.lsArgs;
    return { name: "ls", aliases: [], args: defined({ path: a.path, limit: a.limit }), id: idOf(a) };
  }
  return undefined;
}

function declaredName(name: string, aliases: string[], declared: Set<string>): string {
  if (declared.size === 0 || declared.has(name)) return name;
  for (const alias of aliases) if (declared.has(alias)) return alias;
  return name;
}

function defined(args: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    if (value === undefined || value === null || value === "") continue;
    out[key] = value;
  }
  return out;
}

function callerToolName(a: Record<string, any>, declared: Set<string>): string {
  const candidates = [a.toolName, a.name].filter((v): v is string => typeof v === "string" && v !== "");
  for (const c of candidates) if (declared.has(c)) return c;
  for (const c of candidates) {
    for (const prefix of [`${a.providerIdentifier ?? MCP_PROVIDER}-`, `${MCP_PROVIDER}-`]) {
      if (c.startsWith(prefix) && declared.has(c.slice(prefix.length))) return c.slice(prefix.length);
    }
  }
  return candidates[0] ?? "unknown";
}

/** Rough token count: ~4 ASCII characters per token, one token per other character. */
export function estimateTokens(text: string): number {
  let ascii = 0;
  let other = 0;
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) < 0x80) ascii++;
    else other++;
  }
  return Math.ceil(ascii / 4) + other;
}

function finiteOrUndefined(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}
