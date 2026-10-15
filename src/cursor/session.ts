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
        const presented = presentCallerTool(native, opts.tools);
        push({
          type: "tool_call",
          id: safeToolCallId(native.id),
          name: presented.name,
          args: presented.args,
        });
        // Same as mcp_args: the caller runs the tool and replays the result.
        end({ type: "done", reason: "tool_calls" });
        return;
      }
      if (ex.mcpArgs) {
        const a = ex.mcpArgs;
        const args: Record<string, unknown> = {};
        for (const [key, raw] of Object.entries(a.args ?? {})) args[key] = decodeArgValue(raw);
        const name = callerToolName(a, declaredTools);
        push({
          type: "tool_call",
          id: safeToolCallId(a.toolCallId),
          name,
          args: alignCallerToolArgs(name, args, opts.tools),
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
    return { name: "bash", aliases: ["shell", "Bash"], args: defined({ command: a.command, timeout }), id: idOf(a) };
  }
  if (ex.piReadArgs || ex.readArgs || ex.redactedReadArgs) {
    const a = ex.piReadArgs ?? ex.readArgs ?? ex.redactedReadArgs;
    return { name: "read", aliases: ["Read"], args: defined({ path: a.path, offset: a.offset, limit: a.limit }), id: idOf(a) };
  }
  if (ex.piEditArgs) {
    const a = ex.piEditArgs;
    const edits = Array.isArray(a.edits)
      ? a.edits.map((edit: Record<string, any>) => defined({ oldText: edit.oldText, newText: edit.newText }))
      : [];
    return { name: "edit", aliases: ["Edit"], args: defined({ path: a.path, edits }), id: idOf(a) };
  }
  if (ex.piWriteArgs || ex.writeArgs) {
    const a = ex.piWriteArgs ?? ex.writeArgs;
    return {
      name: "write",
      aliases: ["Write"],
      args: defined({ path: a.path, content: a.content ?? a.fileText }),
      id: idOf(a),
    };
  }
  if (ex.piGrepArgs || ex.grepArgs) {
    const a = ex.piGrepArgs ?? ex.grepArgs;
    return {
      name: "grep",
      aliases: ["Grep"],
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
    return {
      name: "find",
      aliases: ["Glob", "glob"],
      args: defined({ pattern: a.pattern, path: a.path, limit: a.limit }),
      id: idOf(a),
    };
  }
  if (ex.piLsArgs || ex.lsArgs) {
    const a = ex.piLsArgs ?? ex.lsArgs;
    return { name: "ls", aliases: [], args: defined({ path: a.path, limit: a.limit }), id: idOf(a) };
  }
  return undefined;
}

// Cursor and Kimi-backed Claude models often emit Pi/Cursor arg names (`path`,
// `filePath`, `oldString`) for Claude Code tools whose schema says `file_path`.
// Same idea as the Grok bridge: keep the caller's declared names and fields.
// A schema that already uses `path` (Pi read) is left alone.
const CALLER_ARG_ALIASES: Record<string, string[]> = {
  path: ["file_path", "filePath", "target_file", "target_directory"],
  file: ["file_path", "filePath", "path", "target_file"],
  filepath: ["file_path", "filePath", "path", "target_file"],
  filePath: ["file_path", "path", "target_file"],
  file_path: ["filePath", "path", "target_file"],
  oldString: ["old_string"],
  old_string: ["oldString"],
  oldText: ["old_string", "oldString"],
  old_text: ["old_string", "oldString"],
  newString: ["new_string"],
  new_string: ["newString"],
  newText: ["new_string", "newString"],
  new_text: ["new_string", "newString"],
  replaceAll: ["replace_all"],
  replace_all: ["replaceAll"],
  command: ["cmd"],
  cmd: ["command"],
  glob: ["glob_pattern"],
  ignoreCase: ["ignore_case"],
  ignore_case: ["ignoreCase"],
  headLimit: ["head_limit", "limit"],
  head_limit: ["headLimit", "limit"],
};

function editText(edit: Record<string, unknown>, which: "old" | "new"): unknown {
  if (which === "old") return edit.oldText ?? edit.old_text ?? edit.oldString ?? edit.old_string;
  return edit.newText ?? edit.new_text ?? edit.newString ?? edit.new_string;
}

export function alignCallerToolArgs(
  name: string,
  args: Record<string, unknown>,
  tools: McpToolDefinition[],
): Record<string, unknown> {
  const tool = tools.find((item) => item.name === name || item.toolName === name);
  if (!tool) return args;
  let schema: { properties?: Record<string, unknown> } = {};
  try {
    schema = JSON.parse(tool.inputSchemaJson) as { properties?: Record<string, unknown> };
  } catch {
    return args;
  }
  const props = schema.properties;
  if (!props || typeof props !== "object") return args;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    // Pi sends { edits: [{ oldText, newText }] }. Claude Code's Edit is one
    // old_string/new_string pair and has no edits array.
    if (key === "edits" && !Object.prototype.hasOwnProperty.call(props, "edits") && Array.isArray(value)) {
      const first = value.find((item) => item && typeof item === "object") as Record<string, unknown> | undefined;
      if (first) {
        const oldText = editText(first, "old");
        const newText = editText(first, "new");
        if (
          oldText !== undefined &&
          Object.prototype.hasOwnProperty.call(props, "old_string") &&
          !Object.prototype.hasOwnProperty.call(out, "old_string")
        ) {
          out.old_string = oldText;
        }
        if (
          newText !== undefined &&
          Object.prototype.hasOwnProperty.call(props, "new_string") &&
          !Object.prototype.hasOwnProperty.call(out, "new_string")
        ) {
          out.new_string = newText;
        }
      }
      continue;
    }
    if (Object.prototype.hasOwnProperty.call(props, key)) {
      out[key] = value;
      continue;
    }
    const target = (CALLER_ARG_ALIASES[key] ?? []).find((candidate) =>
      Object.prototype.hasOwnProperty.call(props, candidate),
    );
    if (!target) {
      out[key] = value;
      continue;
    }
    // Schema already has the real field, including when the model sent both.
    if (!Object.prototype.hasOwnProperty.call(args, target) && !Object.prototype.hasOwnProperty.call(out, target)) {
      out[target] = value;
    }
  }
  return out;
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

// Cursor's own prompt still says Shell / Read / Grep. Callers declare different
// names for the same capability: Claude Code uses Bash / Read / Write / Edit,
// Pi uses the lowercase names, Codex uses exec_command, and Grok Build uses
// run_terminal_command / read_file / search_replace / list_dir. Map only when
// exactly one declared tool is in the group.
const TOOL_NAME_GROUPS: string[][] = [
  ["bash", "shell", "exec_command", "run_terminal_command", "run_terminal_cmd", "shell_command"],
  ["read", "read_file"],
  ["write", "write_file", "search_replace"],
  ["edit", "strreplace", "str_replace", "search_replace"],
  ["grep"],
  ["glob", "find"],
  ["webfetch", "web_fetch"],
  ["websearch", "web_search"],
  ["todowrite", "todo_write"],
  ["ls", "list_dir"],
];

/**
 * Translate one native exec into the tool name and arguments the caller declared.
 * Codex declares exec_command and no Read/Edit/Write, so a native read or edit
 * becomes one shell command. Grok Build and Claude Code declare the file tools,
 * so those keep their structured arguments.
 */
export function presentCallerTool(
  native: { name: string; aliases: string[]; args: Record<string, unknown> },
  tools: McpToolDefinition[],
): { name: string; args: Record<string, unknown> } {
  const declared = new Set(tools.map((tool) => tool.name).filter((name) => name !== ""));
  let name = resolveCallerToolName(declaredName(native.name, native.aliases, declared), declared);
  let args = native.args;
  if (name.toLowerCase() === "search_replace" && isFileWrite(args)) {
    args = { path: args.path ?? args.file_path ?? args.filePath, old_string: "", new_string: args.content };
  }
  if (declared.size > 0 && !declared.has(name)) {
    const shell = soleDeclared(declared, TOOL_NAME_GROUPS[0] ?? []);
    const command = shell ? shellCommandFor(native.name, native.args) : undefined;
    if (shell && command) {
      name = shell;
      args = { command };
    }
  }
  return { name, args: alignCallerToolArgs(name, args, tools) };
}

function isFileWrite(args: Record<string, unknown>): boolean {
  return (
    typeof args.content === "string" &&
    args.old_string === undefined &&
    args.oldString === undefined &&
    args.edits === undefined
  );
}

function soleDeclared(declared: Set<string>, group: string[]): string | undefined {
  const names = new Set(group);
  const hits = [...declared].filter((name) => names.has(name.toLowerCase()));
  return hits.length === 1 ? hits[0] : undefined;
}

function shellCommandFor(kind: string, args: Record<string, unknown>): string | undefined {
  const name = kind.toLowerCase();
  if (name === "bash" || name === "shell") {
    const command = args.command ?? args.cmd;
    return typeof command === "string" && command !== "" ? command : undefined;
  }
  if (name === "read") return readCommand(args);
  if (name === "grep") return grepCommand(args);
  if (name === "ls") return lsCommand(args);
  if (name === "find" || name === "glob") return findCommand(args);
  if (name === "write") return writeCommand(args);
  if (name === "edit" || name === "strreplace" || name === "str_replace") return editCommand(args);
  return undefined;
}

function shQuote(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : undefined;
}

function readCommand(args: Record<string, unknown>): string | undefined {
  const path = asString(args.path) ?? asString(args.file_path) ?? asString(args.filePath);
  if (!path) return undefined;
  const offset = asNumber(args.offset);
  const limit = asNumber(args.limit);
  if (offset === undefined && limit === undefined) return `cat -- ${shQuote(path)}`;
  const start = Math.max(1, offset ?? 1);
  const end = limit === undefined ? "$" : String(start + Math.max(limit, 0) - 1);
  return `sed -n ${shQuote(`${start},${end}p`)} -- ${shQuote(path)}`;
}

function grepCommand(args: Record<string, unknown>): string | undefined {
  const pattern = asString(args.pattern);
  if (!pattern) return undefined;
  const bits = ["rg", "-n"];
  if (args.ignoreCase === true || args.ignore_case === true) bits.push("-i");
  if (args.literal === true) bits.push("-F");
  const context = asNumber(args.context);
  if (context !== undefined && context > 0) bits.push("-C", String(context));
  const glob = asString(args.glob) ?? asString(args.glob_pattern);
  if (glob) bits.push("-g", shQuote(glob));
  const limit = asNumber(args.limit) ?? asNumber(args.headLimit) ?? asNumber(args.head_limit);
  if (limit !== undefined && limit > 0) bits.push("-m", String(limit));
  bits.push("--", shQuote(pattern));
  const path = asString(args.path);
  if (path) bits.push(shQuote(path));
  return bits.join(" ");
}

function lsCommand(args: Record<string, unknown>): string | undefined {
  const path = asString(args.path) ?? ".";
  let command = `ls -la -- ${shQuote(path)}`;
  const limit = asNumber(args.limit);
  if (limit !== undefined && limit > 0) command += ` | head -n ${limit}`;
  return command;
}

function findCommand(args: Record<string, unknown>): string | undefined {
  const pattern = asString(args.pattern);
  if (!pattern) return undefined;
  const path = asString(args.path) ?? ".";
  return `rg --files -g ${shQuote(pattern)} -- ${shQuote(path)}`;
}

function writeCommand(args: Record<string, unknown>): string | undefined {
  const path = asString(args.path) ?? asString(args.file_path) ?? asString(args.filePath);
  if (!path || typeof args.content !== "string") return undefined;
  return pythonPayload(
    "import base64,json,pathlib,sys\np=json.loads(base64.b64decode(sys.argv[1]))\npath=pathlib.Path(p[\"path\"])\npath.parent.mkdir(parents=True, exist_ok=True)\npath.write_text(p.get(\"content\",\"\"))",
    { path, content: args.content },
  );
}

function editCommand(args: Record<string, unknown>): string | undefined {
  const path = asString(args.path) ?? asString(args.file_path) ?? asString(args.filePath);
  const edits = editPayloads(args);
  if (!path || edits.length === 0) return undefined;
  return pythonPayload(
    "import base64,json,pathlib,sys\np=json.loads(base64.b64decode(sys.argv[1]))\npath=pathlib.Path(p[\"path\"])\ntext=path.read_text()\nfor e in p[\"edits\"]:\n    old=e[\"old\"]\n    new=e[\"new\"]\n    count=text.count(old)\n    if count==0:\n        raise SystemExit(\"old_string not found\")\n    if count>1 and not e.get(\"all\"):\n        raise SystemExit(\"old_string matched more than once\")\n    text=text.replace(old, new, 0 if e.get(\"all\") else 1)\npath.write_text(text)",
    { path, edits },
  );
}

function editPayloads(args: Record<string, unknown>): Array<{ old: string; new: string; all: boolean }> {
  if (Array.isArray(args.edits)) {
    return args.edits.flatMap((item) => {
      if (!item || typeof item !== "object") return [];
      const edit = item as Record<string, unknown>;
      const old = edit.oldText ?? edit.old_text ?? edit.oldString ?? edit.old_string;
      const next = edit.newText ?? edit.new_text ?? edit.newString ?? edit.new_string;
      if (typeof old !== "string" || typeof next !== "string") return [];
      return [{ old, new: next, all: false }];
    });
  }
  const old = args.oldText ?? args.old_text ?? args.oldString ?? args.old_string;
  const next = args.newText ?? args.new_text ?? args.newString ?? args.new_string;
  if (typeof old !== "string" || typeof next !== "string") return [];
  return [{ old, new: next, all: args.replaceAll === true || args.replace_all === true }];
}

function pythonPayload(script: string, payload: unknown): string {
  const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
  return `python3 -c ${shQuote(script)} ${shQuote(encoded)}`;
}

export function resolveCallerToolName(raw: string, declared: Set<string>): string {
  if (!raw || declared.size === 0 || declared.has(raw)) return raw;
  const lower = raw.toLowerCase();
  const sameCase = [...declared].filter((name) => name.toLowerCase() === lower);
  if (sameCase.length === 1) return sameCase[0]!;
  const group = TOOL_NAME_GROUPS.find((item) => item.includes(lower));
  if (!group) return raw;
  const hits = [...declared].filter((name) => group.includes(name.toLowerCase()));
  return hits.length === 1 ? hits[0]! : raw;
}

function callerToolName(a: Record<string, any>, declared: Set<string>): string {
  const candidates = [a.toolName, a.name].filter((v): v is string => typeof v === "string" && v !== "");
  const prefixes = [`${a.providerIdentifier ?? MCP_PROVIDER}-`, `${MCP_PROVIDER}-`];
  const bare = (value: string): string => {
    for (const prefix of prefixes) {
      if (value.startsWith(prefix)) return value.slice(prefix.length);
    }
    return value;
  };
  for (const c of candidates) {
    const resolved = resolveCallerToolName(bare(c), declared);
    if (declared.size === 0 || declared.has(resolved)) return resolved;
  }
  return resolveCallerToolName(bare(candidates[0] ?? "unknown"), declared);
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
