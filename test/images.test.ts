import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import test from "node:test";
import { AgentClientMessage } from "../src/cursor/proto.js";
import { buildUserMessage } from "../src/cursor/session.js";
import { buildTurnInput, type OpenAIChatRequest } from "../src/openai/translate.js";
import type { Config } from "../src/config.js";

const config = { systemAsHistory: true, continuationPrompt: "continue" } as Config;

function png(width: number, height: number, rgb: [number, number, number]): Buffer {
  const row = Buffer.concat([Buffer.from([0]), Buffer.from(rgb)]);
  const raw = Buffer.concat(Array.from({ length: height }, () => row));
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const chunk = (tag: string, data: Buffer): Buffer => {
    const type = Buffer.from(tag);
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([type, data])) >>> 0);
    return Buffer.concat([len, type, data, crc]);
  };
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function crc32(buf: Buffer): number {
  let c = ~0;
  for (const byte of buf) {
    c ^= byte;
    for (let i = 0; i < 8; i++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c;
}

test("active user image becomes selected_images bytes", () => {
  const image = png(1, 1, [0, 255, 0]);
  const body: OpenAIChatRequest = {
    messages: [
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:image/png;base64,${image.toString("base64")}` } },
          { type: "text", text: "color?" },
        ],
      },
    ],
  };
  const input = buildTurnInput(body, config);
  assert.equal(input.userText, "color?");
  assert.equal(input.images.length, 1);
  assert.equal(input.images[0]!.mimeType, "image/png");
  assert.equal(input.images[0]!.width, 1);
  assert.equal(input.images[0]!.height, 1);
  assert.ok(input.images[0]!.data.equals(image));

  const encoded = AgentClientMessage.encode(
    AgentClientMessage.create({
      runRequest: {
        action: { userMessageAction: { userMessage: buildUserMessage(input.userText, input.images, "m1") } },
      },
    }),
  ).finish();
  const decoded = AgentClientMessage.toObject(AgentClientMessage.decode(encoded), {
    defaults: false,
    bytes: String,
  }) as {
    runRequest?: {
      action?: {
        userMessageAction?: {
          userMessage?: { text?: string; selectedContext?: { selectedImages?: Array<{ mimeType?: string; data?: string; dimension?: { width?: number; height?: number } }> } };
        };
      };
    };
  };
  const selected = decoded.runRequest?.action?.userMessageAction?.userMessage?.selectedContext?.selectedImages?.[0];
  assert.equal(decoded.runRequest?.action?.userMessageAction?.userMessage?.text, "color?");
  assert.equal(selected?.mimeType, "image/png");
  assert.equal(selected?.dimension?.width, 1);
  assert.equal(selected?.dimension?.height, 1);
  assert.equal(selected?.data, image.toString("base64"));
});

test("image-only turn is still a user turn", () => {
  const image = png(1, 1, [255, 0, 0]);
  const input = buildTurnInput(
    {
      messages: [{ role: "user", content: [{ type: "input_image", image_url: `data:image/png;base64,${image.toString("base64")}` }] }],
    },
    config,
  );
  assert.equal(input.userText, "");
  assert.equal(input.images.length, 1);
});
