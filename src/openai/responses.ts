/**
 * OpenAI Responses API <-> the chat-completions turn this gateway already runs.
 *
 * Codex CLI (`wire_api = "responses"`) and Grok Build (`api_backend = "responses"`)
 * both post here. The turn itself is unchanged: this module only converts the
 * request in and the completed turn back out.
 */
import { randomUUID } from "node:crypto";
import type { OpenAIChatRequest, OpenAIContentPart, OpenAIMessage, OpenAIToolCall } from "./translate.js";

export interface ResponsesRequest {
  model?: string;
  instructions?: string;
  input?: string | ResponsesItem[];
  tools?: ResponsesTool[];
  stream?: boolean;
  reasoning?: { effort?: string };
  reasoning_effort?: string;
}

export interface ResponsesItem {
  type?: string;
  role?: string;
  content?: unknown;
  name?: string;
  arguments?: unknown;
  call_id?: string;
  id?: string;
  output?: unknown;
  input?: unknown;
}

export interface ResponsesTool {
  type?: string;
  name?: string;
  description?: string;
  parameters?: unknown;
  function?: { name?: string; description?: string; parameters?: unknown };
}

export interface ResponsesTurn {
  text: string;
  toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }>;
  usage: { inputTokens?: number; outputTokens?: number };
  error?: string;
}

export interface ResponsesEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface PreparedResponses {
  request: OpenAIChatRequest;
  customTools: Set<string>;
}

export function responsesToChat(body: ResponsesRequest): PreparedResponses {
  const messages: OpenAIMessage[] = [];
  const instructions = typeof body.instructions === "string" ? body.instructions.trim() : "";
  if (instructions) messages.push({ role: "system", content: instructions });

  const pending: OpenAIToolCall[] = [];
  const callNames = new Map<string, string>();
  const flush = (): void => {
    if (pending.length === 0) return;
    messages.push({ role: "assistant", content: null, tool_calls: pending.splice(0) });
  };

  for (const item of inputItems(body.input)) {
    const type = item.type ?? (item.role ? "message" : "");
    if (type === "function_call" || type === "custom_tool_call") {
      const callId = item.call_id || item.id || `call_${pending.length}`;
      const name = item.name ?? "";
      callNames.set(callId, name);
      const raw = type === "custom_tool_call" ? item.input : item.arguments;
      pending.push({
        id: callId,
        type: "function",
        function: { name, arguments: typeof raw === "string" ? raw : JSON.stringify(raw ?? {}) },
      });
      continue;
    }
    flush();
    if (type === "function_call_output" || type === "custom_tool_call_output") {
      const callId = item.call_id ?? "";
      messages.push({
        role: "tool",
        tool_call_id: callId,
        name: callNames.get(callId) ?? item.name ?? "",
        content: textOfOutput(item.output),
      });
      continue;
    }
    if (type === "message" || item.role) {
      const role = item.role === "assistant" || item.role === "system" || item.role === "developer" ? item.role : "user";
      const content = contentParts(item.content);
      const hasContent = typeof content === "string" ? content.trim() !== "" : (content?.length ?? 0) > 0;
      if (hasContent && content) messages.push({ role, content });
    }
  }
  flush();

  const customTools = new Set<string>();
  const tools: OpenAIChatRequest["tools"] = [];
  for (const tool of body.tools ?? []) {
    const name = tool.name ?? tool.function?.name ?? "";
    if (!name) continue;
    const custom = tool.type === "custom";
    if (custom) customTools.add(name);
    tools.push({
      type: "function",
      function: {
        name,
        description: tool.description ?? tool.function?.description ?? "",
        parameters: custom
          ? {
              type: "object",
              properties: { input: { type: "string" } },
              required: ["input"],
              additionalProperties: false,
            }
          : (tool.parameters ?? tool.function?.parameters ?? { type: "object", properties: {} }),
      },
    });
  }

  return {
    request: {
      model: body.model,
      messages,
      stream: body.stream,
      tools,
      reasoning_effort: body.reasoning_effort ?? body.reasoning?.effort,
    },
    customTools,
  };
}

export function responsesEvents(model: string, turn: ResponsesTurn, customTools: Set<string>): ResponsesEvent[] {
  const id = `resp_${cryptoId()}`;
  const createdAt = Math.floor(Date.now() / 1000);
  const output = outputItems(turn, customTools);
  const usage = usageOf(turn);
  const base = { id, object: "response", created_at: createdAt, model, output: [] as unknown[] };
  if (turn.error) {
    return [
      {
        event: "response.failed",
        data: {
          type: "response.failed",
          response: { ...base, status: "failed", error: { message: turn.error, type: "upstream_error" } },
        },
      },
    ];
  }

  const events: ResponsesEvent[] = [
    { event: "response.created", data: { type: "response.created", response: { ...base, status: "in_progress" } } },
    { event: "response.in_progress", data: { type: "response.in_progress", response: { ...base, status: "in_progress" } } },
  ];

  output.forEach((item, outputIndex) => {
    if (item.type === "message") {
      const text = (item.content[0]?.text as string | undefined) ?? "";
      events.push(
        { event: "response.output_item.added", data: { type: "response.output_item.added", output_index: outputIndex, item: { ...item, status: "in_progress", content: [] } } },
        {
          event: "response.content_part.added",
          data: { type: "response.content_part.added", output_index: outputIndex, content_index: 0, item_id: item.id, part: { type: "output_text", text: "", annotations: [] } },
        },
        { event: "response.output_text.delta", data: { type: "response.output_text.delta", output_index: outputIndex, content_index: 0, item_id: item.id, delta: text } },
        { event: "response.output_text.done", data: { type: "response.output_text.done", output_index: outputIndex, content_index: 0, item_id: item.id, text } },
        { event: "response.content_part.done", data: { type: "response.content_part.done", output_index: outputIndex, content_index: 0, item_id: item.id, part: item.content[0] } },
        { event: "response.output_item.done", data: { type: "response.output_item.done", output_index: outputIndex, item } },
      );
      return;
    }
    if (item.type === "function_call") {
      events.push(
        { event: "response.output_item.added", data: { type: "response.output_item.added", output_index: outputIndex, item: { ...item, arguments: "", status: "in_progress" } } },
        {
          event: "response.function_call_arguments.delta",
          data: { type: "response.function_call_arguments.delta", output_index: outputIndex, item_id: item.id, delta: item.arguments },
        },
        {
          event: "response.function_call_arguments.done",
          data: { type: "response.function_call_arguments.done", output_index: outputIndex, item_id: item.id, arguments: item.arguments },
        },
        { event: "response.output_item.done", data: { type: "response.output_item.done", output_index: outputIndex, item } },
      );
      return;
    }
    events.push(
      { event: "response.output_item.added", data: { type: "response.output_item.added", output_index: outputIndex, item: { ...item, input: "", status: "in_progress" } } },
      {
        event: "response.custom_tool_call_input.delta",
        data: { type: "response.custom_tool_call_input.delta", output_index: outputIndex, item_id: item.id, delta: item.input },
      },
      {
        event: "response.custom_tool_call_input.done",
        data: { type: "response.custom_tool_call_input.done", output_index: outputIndex, item_id: item.id, input: item.input },
      },
      { event: "response.output_item.done", data: { type: "response.output_item.done", output_index: outputIndex, item } },
    );
  });

  events.push({
    event: "response.completed",
    data: { type: "response.completed", response: { ...base, status: "completed", output, usage } },
  });
  return events;
}

export function completedResponse(model: string, turn: ResponsesTurn, customTools: Set<string>): Record<string, unknown> {
  const completed = responsesEvents(model, turn, customTools).find((event) => event.event === "response.completed");
  return (completed?.data.response as Record<string, unknown> | undefined) ?? { status: "failed", error: { message: turn.error ?? "empty response" } };
}

function outputItems(turn: ResponsesTurn, customTools: Set<string>): Array<Record<string, any>> {
  const items: Array<Record<string, any>> = [];
  if (turn.text || turn.toolCalls.length === 0) {
    items.push({
      type: "message",
      id: `msg_${cryptoId()}`,
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: turn.text, annotations: [] }],
    });
  }
  for (const call of turn.toolCalls) {
    const callId = call.id.startsWith("call") ? call.id : `call_${call.id}`;
    if (customTools.has(call.name)) {
      const input = typeof call.args.input === "string" ? call.args.input : JSON.stringify(call.args);
      items.push({ type: "custom_tool_call", id: `ctc_${cryptoId()}`, call_id: callId, name: call.name, input, status: "completed" });
      continue;
    }
    items.push({
      type: "function_call",
      id: `fc_${cryptoId()}`,
      call_id: callId,
      name: call.name,
      arguments: JSON.stringify(call.args ?? {}),
      status: "completed",
    });
  }
  return items;
}

function usageOf(turn: ResponsesTurn): { input_tokens: number; output_tokens: number; total_tokens: number } {
  const input = turn.usage.inputTokens ?? 0;
  const output = turn.usage.outputTokens ?? 0;
  return { input_tokens: input, output_tokens: output, total_tokens: input + output };
}

function inputItems(input: ResponsesRequest["input"]): ResponsesItem[] {
  if (typeof input === "string") return [{ type: "message", role: "user", content: input }];
  return Array.isArray(input) ? input : [];
}

function contentParts(content: unknown): OpenAIMessage["content"] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return textOfOutput(content);
  const parts: OpenAIContentPart[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const item = part as Record<string, unknown>;
    const text = item.text ?? item.output_text ?? item.input_text;
    if (typeof text === "string" && text) parts.push({ type: "text", text });
    const image = item.image_url ?? item.url;
    if (typeof image === "string" && image) parts.push({ type: "image_url", image_url: { url: image } });
    else if (image && typeof image === "object" && typeof (image as { url?: unknown }).url === "string") {
      parts.push({ type: "image_url", image_url: { url: (image as { url: string }).url } });
    }
  }
  return parts;
}

function textOfOutput(output: unknown): string {
  if (typeof output === "string") return output;
  if (Array.isArray(output)) {
    return output
      .map((part) => {
        if (typeof part === "string") return part;
        if (!part || typeof part !== "object") return "";
        const text = (part as { text?: unknown }).text;
        return typeof text === "string" ? text : "";
      })
      .join("");
  }
  if (output === undefined || output === null) return "";
  return JSON.stringify(output);
}

function cryptoId(): string {
  return randomUUID().replaceAll("-", "");
}
