import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_ALLOWED_TOOLS } from "../src/config.js";
import { nativeToolCallFromExec } from "../src/cursor/session.js";
import { isToolHandback, stripHandbackProse } from "../src/openai/translate.js";

test("default allow-list keeps the caller's tool names", () => {
  assert.deepEqual(DEFAULT_ALLOWED_TOOLS, ["mcp_tool_call", "get_mcp_tools_tool_call"]);
});

test("pi bash and native shell both become a bash tool call", () => {
  const pi = nativeToolCallFromExec({ piBashArgs: { command: "echo STEP2", timeout: 5, toolCallId: "abc" } });
  assert.deepEqual(pi, { name: "bash", aliases: ["shell"], args: { command: "echo STEP2", timeout: 5 }, id: "abc" });

  const shell = nativeToolCallFromExec({ execId: "exec-1", shellArgs: { command: "pwd", timeout: 1000 } });
  assert.equal(shell?.name, "bash");
  assert.deepEqual(shell?.args, { command: "pwd" });
  assert.equal(shell?.id, "exec-1");
});

test("pi edit replacements use Pi's oldText/newText fields", () => {
  const edit = nativeToolCallFromExec({
    piEditArgs: { path: "a.ts", edits: [{ oldText: "foo", newText: "bar" }] },
  });
  assert.equal(edit?.name, "edit");
  assert.deepEqual(edit?.args, { path: "a.ts", edits: [{ oldText: "foo", newText: "bar" }] });
});

test("handback text is the model asking the user to continue", () => {
  assert.equal(isToolHandback("下一步：回一句「继续」。我会 SSH 进去。"), true);
  assert.equal(isToolHandback("The shell tool is unavailable."), true);
  assert.equal(isToolHandback("`bash` 通道又被掐断了。"), true);
  assert.equal(isToolHandback("Shell 工具当前不可用，所以 echo STEP2 没能跑起来。"), true);
  assert.equal(isToolHandback("VM 108 is openclaw-vm at 192.168.124.208. Rules are in /etc/mihomo/config.yaml."), false);
});

test("bash-cut prose is removed and the real finding stays", () => {
  const raw =
    "先在 PVE 上对上 192.168.124.208。`bash` 通道又被掐断了。改用另一条命令通道直接 SSH。\nVM 108 is openclaw-vm. The shell tool is unavailable.";
  assert.equal(
    stripHandbackProse(raw),
    "先在 PVE 上对上 192.168.124.208。改用另一条命令通道直接 SSH。\nVM 108 is openclaw-vm.",
  );
  assert.equal(
    stripHandbackProse("STEP2 没跑成，再执行一次。Shell 工具当前不可用，所以 echo 没能跑起来。"),
    "",
  );
});

test("request-context exec is not a caller tool", () => {
  assert.equal(nativeToolCallFromExec({ requestContextArgs: {} }), undefined);
});
