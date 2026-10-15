import assert from "node:assert/strict";
import test from "node:test";
import { responsesEvents, responsesToChat } from "../src/openai/responses.js";

test("a Codex responses request becomes chat messages and function tools", () => {
  const prepared = responsesToChat({
    model: "gpt-5.5",
    instructions: "follow the repo",
    stream: true,
    reasoning: { effort: "high" },
    input: [
      { type: "message", role: "user", content: [{ type: "input_text", text: "list the files" }] },
      { type: "function_call", name: "exec_command", arguments: "{\"cmd\":\"ls\"}", call_id: "call_1" },
      { type: "function_call_output", call_id: "call_1", output: "a.ts\n" },
    ],
    tools: [
      {
        type: "function",
        name: "exec_command",
        description: "run",
        parameters: { type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"] },
      },
      { type: "custom", name: "apply_patch", description: "patch" },
    ],
  });
  assert.equal(prepared.request.model, "gpt-5.5");
  assert.equal(prepared.request.reasoning_effort, "high");
  assert.deepEqual(
    prepared.request.messages?.map((message) => message.role),
    ["system", "user", "assistant", "tool"],
  );
  assert.equal(prepared.request.messages?.[2]?.tool_calls?.[0]?.function.name, "exec_command");
  assert.equal(prepared.request.messages?.[3]?.content, "a.ts\n");
  assert.deepEqual(
    prepared.request.tools?.map((tool) => tool.function?.name),
    ["exec_command", "apply_patch"],
  );
  assert.equal(prepared.customTools.has("apply_patch"), true);
});

test("responses output emits a function call Codex can execute", () => {
  const events = responsesEvents(
    "gpt-5.5",
    { text: "", toolCalls: [{ id: "call_1", name: "exec_command", args: { cmd: "ls" } }], usage: { inputTokens: 3, outputTokens: 4 } },
    new Set(),
  );
  const names = events.map((event) => event.event);
  assert.ok(names.includes("response.created"));
  assert.ok(names.includes("response.function_call_arguments.delta"));
  assert.equal(names.at(-1), "response.completed");
  const done = events.find((event) => event.event === "response.function_call_arguments.done");
  assert.equal(done?.data.arguments, "{\"cmd\":\"ls\"}");
  const completed = events.at(-1)?.data.response as { output: Array<{ type: string; name?: string }>; usage: { total_tokens: number } };
  assert.equal(completed.output[0]?.type, "function_call");
  assert.equal(completed.output[0]?.name, "exec_command");
  assert.equal(completed.usage.total_tokens, 7);
});

test("a custom apply_patch call stays freeform text", () => {
  const events = responsesEvents(
    "gpt-5.5",
    { text: "done", toolCalls: [{ id: "call_2", name: "apply_patch", args: { input: "*** Begin Patch\n*** End Patch\n" } }], usage: {} },
    new Set(["apply_patch"]),
  );
  const done = events.find((event) => event.event === "response.custom_tool_call_input.done");
  assert.equal(done?.data.input, "*** Begin Patch\n*** End Patch\n");
  const completed = events.at(-1)?.data.response as { output: Array<{ type: string }> };
  assert.deepEqual(
    completed.output.map((item) => item.type),
    ["message", "custom_tool_call"],
  );
});
