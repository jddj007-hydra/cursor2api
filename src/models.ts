/**
 * Model catalogue.
 *
 * The list comes from upstream `agent.v1.AgentService/GetUsableModels` (198 entries
 * on a Pro account today) and is cached in-process. Client-visible names carry the
 * configured prefix, with one wrinkle: a handful of upstream ids already start with
 * `cursor-` (`cursor-grok-4.6-high`), so prefixing must not double up.
 */
import protobuf from "protobufjs";
import path from "node:path";
import type { Config } from "./config.js";
import { protoDir } from "./cursor/proto.js";
import { unaryCall } from "./cursor/unary.js";

const agentRoot = protobuf.loadSync(path.join(protoDir, "agent.proto"));
const GetUsableModelsRequest = agentRoot.lookupType("agent.v1.GetUsableModelsRequest");
const GetUsableModelsResponse = agentRoot.lookupType("agent.v1.GetUsableModelsResponse");

/** Used until the first successful upstream fetch, and if the RPC ever fails. */
export const FALLBACK_MODEL_IDS = [
  "auto",
  "claude-opus-5-thinking-high",
  "claude-sonnet-5-thinking-high",
  "claude-fable-5-thinking-high",
  "claude-opus-4-8-thinking-high",
  "composer-2.5",
  "gpt-5.6-sol-high",
  "gpt-5.6-luna-high",
  "gpt-5.3-codex",
  "gpt-5.2",
  "cursor-grok-4.6-high",
  "gemini-3.7-flash-high",
];

const CACHE_TTL_MS = 10 * 60 * 1000;

let cache: { ids: string[]; at: number } | null = null;

export function clientName(upstreamId: string, prefix: string): string {
  if (!prefix) return upstreamId;
  return upstreamId.startsWith(prefix) ? upstreamId : `${prefix}${upstreamId}`;
}

/**
 * Resolve a client-visible model name back to an upstream id. Falls back to the
 * name as given so a caller can always address a model the catalogue hasn't seen.
 */
export function upstreamId(name: string, prefix: string, known: readonly string[]): string {
  const trimmed = name.trim();
  if (known.includes(trimmed)) return trimmed;
  if (prefix && trimmed.startsWith(prefix)) {
    const stripped = trimmed.slice(prefix.length);
    if (known.includes(stripped)) return stripped;
    // Unknown to the catalogue: prefer the stripped form, since that is what a
    // prefixed client name means, unless stripping empties it.
    if (stripped) return stripped;
  }
  return trimmed;
}

const EFFORT_SUFFIX = /-(low|medium|high|xhigh|max)(-fast)?$/;

/** Requested reasoning effort → upstream variant suffixes to try, in order. */
const EFFORT_VARIANTS: Record<string, string[]> = {
  none: ["low"],
  minimal: ["low"],
  low: ["low"],
  medium: ["medium"],
  high: ["high"],
  xhigh: ["xhigh", "high"],
  "x-high": ["xhigh", "high"],
  max: ["max", "xhigh", "high"],
};

function globToRegExp(glob: string): RegExp {
  const escaped = glob.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  return new RegExp(`^${escaped}$`, "i");
}

/**
 * Cursor encodes reasoning effort in the model id (`grok-4.7-low` … `-xhigh`) and
 * ignores any effort parameter. For models matching `patterns`, swap the id's
 * effort suffix for the one the caller asked for, keeping a `-fast` tail, when that
 * variant exists upstream. Gateways in front of us may fill in a default effort for
 * callers that sent none, which is why this is limited to an opt-in model set.
 */
export function applyRequestedEffort(
  model: string,
  effort: string | undefined,
  known: readonly string[],
  patterns: readonly string[],
  prefix: string,
): string {
  const variants = effort ? EFFORT_VARIANTS[effort.trim().toLowerCase()] : undefined;
  if (!variants) return model;
  const bare = prefix && model.startsWith(prefix) ? model.slice(prefix.length) : model;
  if (!patterns.some((p) => globToRegExp(p).test(model) || globToRegExp(p).test(bare))) return model;
  const match = EFFORT_SUFFIX.exec(model);
  if (!match) return model;
  const stem = model.slice(0, match.index);
  for (const v of variants) {
    const candidate = `${stem}-${v}${match[2] ?? ""}`;
    if (known.includes(candidate)) return candidate;
  }
  return model;
}

export async function listModelIds(token: string, config: Config): Promise<string[]> {
  if (cache && Date.now() - cache.at < CACHE_TTL_MS) return cache.ids;
  try {
    const body = GetUsableModelsRequest.encode(GetUsableModelsRequest.create({})).finish();
    const res = await unaryCall("/agent.v1.AgentService/GetUsableModels", body, token, config);
    if (res.payload) {
      const decoded = GetUsableModelsResponse.toObject(GetUsableModelsResponse.decode(res.payload), {
        defaults: false,
        enums: String,
        longs: Number,
      }) as { models?: Array<{ modelId?: string }> };
      const ids = (decoded.models ?? [])
        .map((m) => m.modelId)
        .filter((id): id is string => typeof id === "string" && id.length > 0);
      if (ids.length > 0) {
        cache = { ids, at: Date.now() };
        return ids;
      }
    }
  } catch {
    /* fall through to the static list */
  }
  const ids = cache?.ids ?? FALLBACK_MODEL_IDS;
  cache = { ids, at: Date.now() };
  return ids;
}

export interface OpenAIModelEntry {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

export function toModelList(ids: readonly string[], prefix: string): OpenAIModelEntry[] {
  const created = Math.floor(Date.now() / 1000);
  const seen = new Set<string>();
  const out: OpenAIModelEntry[] = [];
  for (const id of ids) {
    const name = clientName(id, prefix);
    if (seen.has(name)) continue;
    seen.add(name);
    out.push({ id: name, object: "model", created, owned_by: "cursor" });
  }
  return out;
}
