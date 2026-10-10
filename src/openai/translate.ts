/**
 * OpenAI chat-completions <-> Cursor agent translation.
 *
 * Cursor's server builds the model prompt itself and only accepts conversation
 * history from the client, so a caller's `system` message is mapped onto a history
 * exchange (an instruction turn plus a short assistant acknowledgement) rather than
 * a `system` role entry. Set CURSOR_DIRECT_SYSTEM_AS_HISTORY=false to send the
 * system role verbatim instead.
 */
import { MCP_PROVIDER, type McpToolDefinition } from "../cursor/session.js";
import type { Config } from "../config.js";

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface OpenAIContentPart {
  type?: string;
  text?: string;
  image_url?: string | { url?: string };
  url?: string;
  source?: { type?: string; media_type?: string; data?: string };
}

export interface OpenAIMessage {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  content?: string | Array<OpenAIContentPart | string> | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  name?: string;
}

/** Raster image extracted from an OpenAI/Anthropic content part. */
export interface InlineImage {
  mimeType: string;
  data: Buffer;
  path: string;
  width?: number;
  height?: number;
}

const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_IMAGES = 8;

export interface OpenAIChatRequest {
  model?: string;
  messages?: OpenAIMessage[];
  stream?: boolean;
  tools?: Array<{ type?: string; function?: { name?: string; description?: string; parameters?: unknown } }>;
  tool_choice?: unknown;
  reasoning_effort?: string;
  reasoning?: { effort?: string };
}

export interface TurnInput {
  rootMessages: unknown[];
  userText: string;
  /** Images on the active user turn. Cursor sees these as selected_images. */
  images: InlineImage[];
  tools: McpToolDefinition[];
}

/**
 * Cursor's API accepts no client system role, so a caller's `system` message is
 * delivered as the opening exchange of the conversation. It is labelled for what it
 * is — configuration from the operator of this gateway — rather than dressed up to
 * look like a platform system prompt. The model may still decline parts of it; that
 * is passed back to the caller as-is.
 */
const SYSTEM_LEAD_IN =
  "Configuration for this session, set by the operator of this API gateway:";
const SYSTEM_ACK = "Understood. I will follow that configuration for the rest of this conversation.";

export function textOf(content: OpenAIMessage["content"]): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : (part?.text ?? "")))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

/**
 * Pull inline raster images out of a content array.
 * Accepts OpenAI `image_url` (string or `{url}`) and Anthropic `source.base64`.
 * Remote http(s) URLs are left behind: Cursor's selected_images field wants bytes.
 */
export function imagesOf(content: OpenAIMessage["content"]): InlineImage[] {
  if (!Array.isArray(content)) return [];
  const out: InlineImage[] = [];
  for (const part of content) {
    if (typeof part === "string") continue;
    const image = imageFromPart(part);
    if (!image) continue;
    out.push({ ...image, path: imagePath(image.mimeType, out.length) });
    if (out.length >= MAX_IMAGES) break;
  }
  return out;
}

function imageFromPart(part: OpenAIContentPart | undefined): Omit<InlineImage, "path"> | undefined {
  if (!part) return;
  const type = part.type ?? "";
  if (type !== "image_url" && type !== "input_image" && type !== "image") return;
  const uri = imageUriOf(part);
  if (!uri) return;
  return decodeDataImage(uri);
}

function imageUriOf(part: OpenAIContentPart): string {
  const direct = part.image_url ?? part.url;
  if (typeof direct === "string") return direct;
  if (direct && typeof direct.url === "string") return direct.url;
  const source = part.source;
  if (source?.type === "base64" && source.data) {
    return `data:${source.media_type || "image/png"};base64,${source.data}`;
  }
  return "";
}

function decodeDataImage(uri: string): Omit<InlineImage, "path"> | undefined {
  const match = /^data:(image\/[a-z0-9.+-]+);base64,([a-z0-9+/=\s]+)$/i.exec(uri.trim());
  if (!match) return;
  const mimeType = match[1]!.toLowerCase() === "image/jpg" ? "image/jpeg" : match[1]!.toLowerCase();
  if (!["image/png", "image/jpeg", "image/gif", "image/webp"].includes(mimeType)) return;
  let data: Buffer;
  try {
    data = Buffer.from(match[2]!.replace(/\s/g, ""), "base64");
  } catch {
    return;
  }
  if (data.length === 0 || data.length > MAX_IMAGE_BYTES) return;
  const size = imageSize(data, mimeType);
  return { mimeType, data, ...size };
}

function imagePath(mimeType: string, index: number): string {
  const ext = mimeType === "image/jpeg" ? "jpg" : mimeType.slice("image/".length);
  return index === 0 ? `image.${ext}` : `image-${index + 1}.${ext}`;
}

function imageSize(data: Buffer, mimeType: string): { width?: number; height?: number } {
  if (mimeType === "image/png" && data.length >= 24 && data[0] === 0x89 && data.toString("ascii", 1, 4) === "PNG") {
    return { width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
  }
  if (mimeType === "image/gif" && data.length >= 10 && data.toString("ascii", 0, 3) === "GIF") {
    return { width: data.readUInt16LE(6), height: data.readUInt16LE(8) };
  }
  if (mimeType === "image/jpeg") {
    let i = 2;
    while (i + 9 < data.length) {
      if (data[i] !== 0xff) {
        i += 1;
        continue;
      }
      const marker = data[i + 1]!;
      if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2) {
        return { height: data.readUInt16BE(i + 5), width: data.readUInt16BE(i + 7) };
      }
      const len = data.readUInt16BE(i + 2);
      if (len < 2) break;
      i += 2 + len;
    }
  }
  return {};
}

const HANDBACK_PROSE =
  /回一句|工具通道|被掐断|bash\s*通道|工具调用又|工具调用失败|没有发出去|没跑成|没有执行成功|没能跑起来|工具当前不可用|Shell\s*工具[^。\n]{0,24}不可用|tool is unavailable|tools are unavailable|tool not available|shell tool is failing|shell tool is unavailable/i;

/** Model gave up and handed the turn back to the user instead of calling the next tool. */
export function isToolHandback(text: string): boolean {
  return HANDBACK_PROSE.test(text);
}

/**
 * Drop sentences that claim the tool channel died. Those lines are the model
 * copying an earlier stop; the tool result in the same history is the real outcome.
 * Leaving them in makes the next turn repeat "bash 被掐断" and refuse to call bash.
 */
export function stripHandbackProse(text: string): string {
  const kept = text
    .split("\n")
    .map((line) =>
      line
        .split(/(?<=[。！？])|(?<=\.)\s+/)
        .filter((sentence) => sentence.trim() && !HANDBACK_PROSE.test(sentence))
        .join("")
        .trim(),
    )
    .filter((line) => line);
  return kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

export function buildTurnInput(body: OpenAIChatRequest, config: Config): TurnInput {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const rootMessages: unknown[] = [];

  // The active user message is the trailing user turn, if the caller ended on one.
  // Mid-agent-loop requests end on a tool result and get a resume action instead.
  let activeIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const role = messages[i]?.role;
    if (role === "user" || role === "developer") {
      activeIndex = i;
      break;
    }
    if (role === "tool" || role === "assistant") break;
  }

  for (let i = 0; i < messages.length; i++) {
    if (i === activeIndex) continue;
    const msg = messages[i];
    if (!msg) continue;

    if (msg.role === "system" || msg.role === "developer") {
      const text = textOf(msg.content);
      if (!text) continue;
      if (config.systemAsHistory) {
        rootMessages.push({
          role: "user",
          content: [{ type: "text", text: `${SYSTEM_LEAD_IN}\n\n${text}` }],
        });
        rootMessages.push({ role: "assistant", content: [{ type: "text", text: SYSTEM_ACK }] });
      } else {
        rootMessages.push({ role: "system", content: text });
      }
      continue;
    }

    if (msg.role === "user") {
      const text = textOf(msg.content);
      if (text) rootMessages.push({ role: "user", content: [{ type: "text", text }] });
      continue;
    }

    if (msg.role === "assistant") {
      const parts: unknown[] = [];
      const text = stripHandbackProse(textOf(msg.content));
      if (text) parts.push({ type: "text", text });
      for (const call of msg.tool_calls ?? []) {
        parts.push({
          type: "tool-call",
          toolCallId: call.id,
          toolName: call.function?.name ?? "",
          args: parseArgs(call.function?.arguments),
        });
      }
      if (parts.length > 0) rootMessages.push({ role: "assistant", content: parts });
      continue;
    }

    if (msg.role === "tool") {
      // Emit even when empty: the matching assistant `tool-call` is already in
      // history, and dropping the pair would replay an orphaned call.
      rootMessages.push({
        role: "tool",
        id: msg.tool_call_id ?? "",
        content: [
          {
            type: "tool-result",
            toolName: msg.name ?? "",
            toolCallId: msg.tool_call_id ?? "",
            result: textOf(msg.content),
          },
        ],
      });
    }
  }

  // A stateless caller mid-agent-loop ends on a tool result. There is no server-side
  // conversation to resume, so drive the follow-up with a continuation turn instead;
  // everything the model needs is already in `rootMessages`.
  const endsOnToolResult = messages.length > 0 && messages[messages.length - 1]?.role === "tool";
  const userText =
    activeIndex >= 0
      ? textOf(messages[activeIndex]?.content)
      : endsOnToolResult
        ? config.continuationPrompt
        : "";
  const images = activeIndex >= 0 ? imagesOf(messages[activeIndex]?.content) : [];

  return { rootMessages, userText, images, tools: buildTools(body.tools) };
}

function parseArgs(raw: string | undefined): unknown {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { _raw: raw };
  }
}

export function buildTools(tools: OpenAIChatRequest["tools"]): McpToolDefinition[] {
  if (!Array.isArray(tools)) return [];
  const out: McpToolDefinition[] = [];
  for (const tool of tools) {
    const fn = tool?.function;
    const name = fn?.name;
    if (!name) continue;
    out.push({
      name,
      toolName: name,
      providerIdentifier: MCP_PROVIDER,
      description: fn?.description ?? "",
      inputSchemaJson: JSON.stringify(fn?.parameters ?? { type: "object", properties: {} }),
    });
  }
  return out;
}
