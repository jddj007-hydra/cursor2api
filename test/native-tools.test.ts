import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_ALLOWED_TOOLS } from "../src/config.js";
import {
  alignCallerToolArgs,
  nativeToolCallFromExec,
  presentCallerTool,
  resolveCallerToolName,
  type McpToolDefinition,
} from "../src/cursor/session.js";
import { isToolHandback, stripHandbackProse } from "../src/openai/translate.js";

test("default allow-list exposes shell, read, grep, and the Pi tools", () => {
  assert.deepEqual(DEFAULT_ALLOWED_TOOLS, [
    "mcp_tool_call",
    "get_mcp_tools_tool_call",
    "shell_tool_call",
    "read_tool_call",
    "grep_tool_call",
    "ls_tool_call",
    "pi_bash_tool_call",
    "pi_read_tool_call",
    "pi_edit_tool_call",
    "pi_write_tool_call",
    "pi_grep_tool_call",
    "pi_find_tool_call",
    "pi_ls_tool_call",
  ]);
});

test("pi bash and native shell both become a bash tool call", () => {
  const pi = nativeToolCallFromExec({ piBashArgs: { command: "echo STEP2", timeout: 5, toolCallId: "abc" } });
  assert.deepEqual(pi, { name: "bash", aliases: ["shell", "Bash"], args: { command: "echo STEP2", timeout: 5 }, id: "abc" });

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

test("claude code Read path is rewritten to file_path; Pi path stays", () => {
  const claudeRead: McpToolDefinition = {
    name: "Read",
    toolName: "Read",
    providerIdentifier: "openai",
    description: "",
    inputSchemaJson: JSON.stringify({
      type: "object",
      properties: { file_path: { type: "string" }, offset: { type: "integer" } },
      required: ["file_path"],
    }),
  };
  const piRead: McpToolDefinition = {
    name: "read",
    toolName: "read",
    providerIdentifier: "openai",
    description: "",
    inputSchemaJson: JSON.stringify({ type: "object", properties: { path: { type: "string" } } }),
  };
  assert.deepEqual(alignCallerToolArgs("Read", { path: "/tmp/a.py", offset: 1 }, [claudeRead]), {
    file_path: "/tmp/a.py",
    offset: 1,
  });
  assert.deepEqual(alignCallerToolArgs("Read", { file_path: "/keep", path: "/drop" }, [claudeRead]), {
    file_path: "/keep",
  });
  assert.deepEqual(alignCallerToolArgs("read", { path: "src/a.ts" }, [piRead]), { path: "src/a.ts" });
  assert.deepEqual(
    alignCallerToolArgs("Edit", { filePath: "a.py", oldString: "a", newString: "b", replaceAll: false }, [
      {
        name: "Edit",
        toolName: "Edit",
        providerIdentifier: "openai",
        description: "",
        inputSchemaJson: JSON.stringify({
          type: "object",
          properties: {
            file_path: { type: "string" },
            old_string: { type: "string" },
            new_string: { type: "string" },
            replace_all: { type: "boolean" },
          },
        }),
      },
    ]),
    { file_path: "a.py", old_string: "a", new_string: "b", replace_all: false },
  );
});

test("Pi edits become Claude old_string/new_string and stay an array for Pi", () => {
  const claudeEdit: McpToolDefinition = {
    name: "Edit",
    toolName: "Edit",
    providerIdentifier: "openai",
    description: "",
    inputSchemaJson: JSON.stringify({
      type: "object",
      properties: {
        file_path: { type: "string" },
        old_string: { type: "string" },
        new_string: { type: "string" },
      },
      required: ["file_path", "old_string", "new_string"],
    }),
  };
  const piEdit: McpToolDefinition = {
    name: "edit",
    toolName: "edit",
    providerIdentifier: "openai",
    description: "",
    inputSchemaJson: JSON.stringify({
      type: "object",
      properties: {
        path: { type: "string" },
        edits: { type: "array" },
      },
    }),
  };
  const edits = [{ oldText: "foo", newText: "bar" }];
  assert.deepEqual(alignCallerToolArgs("Edit", { path: "a.py", edits }, [claudeEdit]), {
    file_path: "a.py",
    old_string: "foo",
    new_string: "bar",
  });
  assert.deepEqual(alignCallerToolArgs("edit", { path: "a.py", edits }, [piEdit]), {
    path: "a.py",
    edits,
  });
});

test("find maps to Glob only when that is the declared tool", () => {
  assert.equal(resolveCallerToolName("find", new Set(["Glob"])), "Glob");
  assert.equal(resolveCallerToolName("find", new Set(["find", "ls"])), "find");
});

test("cursor Shell/Read names map onto whichever tool the caller declared", () => {
  const claude = new Set(["Bash", "Read", "Write", "Edit", "Grep"]);
  assert.equal(resolveCallerToolName("Shell", claude), "Bash");
  assert.equal(resolveCallerToolName("shell", claude), "Bash");
  assert.equal(resolveCallerToolName("Read", claude), "Read");
  assert.equal(resolveCallerToolName("str_replace", claude), "Edit");
  assert.equal(resolveCallerToolName("WebFetch", claude), "WebFetch");

  const pi = new Set(["bash", "read", "edit", "write"]);
  assert.equal(resolveCallerToolName("Shell", pi), "bash");
  assert.equal(resolveCallerToolName("Read", pi), "read");
  assert.equal(resolveCallerToolName("Bash", pi), "bash");

  const both = new Set(["shell", "Bash"]);
  assert.equal(resolveCallerToolName("Shell", both), "shell");
  assert.equal(resolveCallerToolName("NotATool", claude), "NotATool");
});

test("dynamic-tool handback is stripped", () => {
  assert.equal(isToolHandback("The `Read` tool isn't available in this session."), true);
  assert.equal(isToolHandback("`Write` and `Bash` aren't available as direct tools."), true);
  assert.equal(
    stripHandbackProse("Fix is in calc.py. I can only reach tools through GetDynamicTools and CallDynamicTool."),
    "Fix is in calc.py.",
  );
  assert.equal(
    stripHandbackProse(
      "I'm not going to continue on that instruction alone. The shell tool isn't available directly, so I'll use the dynamic tool gateway. No shell or command tool is exposed in this session.",
    ),
    "",
  );
});

test("request-context exec is not a caller tool", () => {
  assert.equal(nativeToolCallFromExec({ requestContextArgs: {} }), undefined);
});

function tool(name: string, properties: Record<string, unknown>, required: string[] = []): McpToolDefinition {
  return {
    name,
    toolName: name,
    providerIdentifier: "openai",
    description: "",
    inputSchemaJson: JSON.stringify({ type: "object", properties, required, additionalProperties: false }),
  };
}

test("Codex exec_command and Grok run_terminal_command receive the shell command", () => {
  const codex = [tool("exec_command", { cmd: { type: "string" }, workdir: { type: "string" } }, ["cmd"])];
  const grok = [tool("run_terminal_command", { command: { type: "string" }, block_until_ms: { type: "number" } }, ["command"])];
  const shell = nativeToolCallFromExec({ execId: "exec-1", shellArgs: { command: "ls" } });
  assert.ok(shell);
  assert.deepEqual(presentCallerTool(shell, codex), { name: "exec_command", args: { cmd: "ls" } });
  assert.deepEqual(presentCallerTool(shell, grok), { name: "run_terminal_command", args: { command: "ls" } });
  assert.equal(resolveCallerToolName("bash", new Set(["exec_command", "run_terminal_command"])), "bash");
});

test("Grok file tools keep structured arguments", () => {
  const grok = [
    tool("read_file", { target_file: { type: "string" }, offset: { type: "number" }, limit: { type: "number" } }, ["target_file"]),
    tool("search_replace", { file_path: { type: "string" }, old_string: { type: "string" }, new_string: { type: "string" }, replace_all: { type: "boolean" } }, ["file_path", "old_string", "new_string"]),
    tool("write", { file_path: { type: "string" }, content: { type: "string" } }, ["file_path", "content"]),
    tool("list_dir", { target_directory: { type: "string" } }, ["target_directory"]),
    tool("grep", { pattern: { type: "string" }, path: { type: "string" }, glob_pattern: { type: "string" } }, ["pattern"]),
  ];
  const read = nativeToolCallFromExec({ readArgs: { path: "src/a.ts", offset: 2, limit: 10 } });
  const edit = nativeToolCallFromExec({ piEditArgs: { path: "src/a.ts", edits: [{ oldText: "foo", newText: "bar" }] } });
  const write = nativeToolCallFromExec({ writeArgs: { path: "src/b.ts", fileText: "hello" } });
  const ls = nativeToolCallFromExec({ lsArgs: { path: "/tmp" } });
  const grep = nativeToolCallFromExec({ grepArgs: { pattern: "foo", path: "src", glob: "*.ts" } });
  assert.ok(read && edit && write && ls && grep);
  assert.deepEqual(presentCallerTool(read, grok), { name: "read_file", args: { target_file: "src/a.ts", offset: 2, limit: 10 } });
  assert.deepEqual(presentCallerTool(edit, grok), {
    name: "search_replace",
    args: { file_path: "src/a.ts", old_string: "foo", new_string: "bar" },
  });
  assert.deepEqual(presentCallerTool(write, grok), { name: "write", args: { file_path: "src/b.ts", content: "hello" } });
  assert.deepEqual(presentCallerTool(ls, grok), { name: "list_dir", args: { target_directory: "/tmp" } });
  assert.deepEqual(presentCallerTool(grep, grok), { name: "grep", args: { pattern: "foo", path: "src", glob_pattern: "*.ts" } });
});

test("Grok read_file that already uses path is left alone", () => {
  const read = nativeToolCallFromExec({ readArgs: { path: "src/main.rs" } });
  assert.ok(read);
  assert.deepEqual(presentCallerTool(read, [tool("read_file", { path: { type: "string" } }, ["path"])]), {
    name: "read_file",
    args: { path: "src/main.rs" },
  });
});

test("a Grok toolset with only search_replace accepts a new file as an empty old_string", () => {
  const write = nativeToolCallFromExec({ piWriteArgs: { path: "src/new.ts", content: "export {}" } });
  assert.ok(write);
  assert.deepEqual(
    presentCallerTool(write, [
      tool("search_replace", { file_path: { type: "string" }, old_string: { type: "string" }, new_string: { type: "string" } }, ["file_path", "old_string", "new_string"]),
    ]),
    { name: "search_replace", args: { file_path: "src/new.ts", old_string: "", new_string: "export {}" } },
  );
});

test("Codex file tools become one exec_command", () => {
  const codex = [tool("exec_command", { cmd: { type: "string" } }, ["cmd"]), tool("write_stdin", { session_id: { type: "number" } })];
  const read = nativeToolCallFromExec({ readArgs: { path: "src/a.ts", offset: 10, limit: 5 } });
  const edit = nativeToolCallFromExec({ piEditArgs: { path: "src/a.ts", edits: [{ oldText: "foo", newText: "bar" }] } });
  assert.ok(read && edit);
  const readCall = presentCallerTool(read, codex);
  assert.equal(readCall.name, "exec_command");
  assert.equal(readCall.args.cmd, "sed -n '10,14p' -- 'src/a.ts'");
  const editCall = presentCallerTool(edit, codex);
  assert.equal(editCall.name, "exec_command");
  assert.equal(typeof editCall.args.cmd, "string");
  const command = String(editCall.args.cmd);
  assert.match(command, /^python3 -c /);
  const encoded = command.split(" ").at(-1)?.replaceAll("'", "");
  assert.ok(encoded);
  assert.deepEqual(JSON.parse(Buffer.from(encoded, "base64").toString("utf8")), {
    path: "src/a.ts",
    edits: [{ old: "foo", new: "bar", all: false }],
  });
});
