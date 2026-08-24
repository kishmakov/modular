# Mojo LLDB debugging plan

The quickest reliable route is to use `mojo debug`, which launches the bundled
`mojo-lldb` and loads the Mojo plugin automatically. A system `lldb` alone does
not provide the full Mojo debugging experience.

The current shell has Pixi, but no `mojo`, `lldb`, or prebuilt repository
debugger artifacts, so start with the SDK-backed smoke test.

## 1. Install the repository's Mojo Pixi environment

```bash
cd /home/kishmakov/Repos/modular/mojo
pixi install
```

Verify both Mojo and the bundled debugger:

```bash
pixi run mojo --version
pixi run mojo debug -X --version
pixi run mojo debug --help
```

`-X` passes the following argument directly to LLDB.

## 2. Inspect what `mojo debug` will launch

Use the repository's formatter test program:

```bash
pixi run mojo debug --dry-run \
  ../KGEN/test/mojo-debug/Inputs/dict.mojo
```

For a source file, `mojo debug` effectively launches `mojo run` with `-O0` and
full debug information. The repository verifies that behavior in
`KGEN/test/mojo-debug/mojo-debug-cli.lldb`.

## 3. Start an interactive LLDB session

```bash
pixi run mojo debug \
  ../KGEN/test/mojo-debug/Inputs/dict.mojo
```

At the `(lldb)` prompt, run these commands in order:

```text
breakpoint set --file dict.mojo --line 20
run
frame variable d
thread backtrace
next
continue
quit
```

At line 20, `d` should contain three entries. The important check is that
`frame variable d` uses the Mojo formatter and prints something beginning with:

```text
d = (size 3)
```

This scenario comes from
`KGEN/test/mojo-debug/dict-formatter.lldb` and its source
`KGEN/test/mojo-debug/Inputs/dict.mojo`.

Prefer `frame variable`---abbreviated `v`---over `expr` or `p`. The repository
notes that expression evaluation may attempt unsupported JIT compilation.
Commands such as `frame variable d`, `frame variable *ptr`, and
`frame variable object.field` are the reliable ways to inspect values.

## 4. Repeat with a precompiled binary

This confirms ordinary DWARF debugging rather than the `mojo run` JIT path:

```bash
pixi run mojo build \
  -O0 --debug-level=full \
  ../KGEN/test/mojo-debug/Inputs/dict.mojo \
  -o /tmp/mojo-lldb-dict

pixi run mojo debug /tmp/mojo-lldb-dict
```

Then repeat:

```text
breakpoint set --file dict.mojo --line 20
run
frame variable d
thread backtrace
continue
quit
```

For precompiled binaries, `-O0 --debug-level=full` is important. Source-file
debugging adds these defaults automatically.

## 5. Test the debugger built from this checkout

If the goal is to exercise changes in `KGEN/lib/MojoLLDB`, build the in-tree
tool instead of using Pixi's packaged version:

```bash
cd /home/kishmakov/Repos/modular

./bazelw build //KGEN:mojo

./bazelw run //KGEN:mojo -- \
  debug --dry-run KGEN/test/mojo-debug/Inputs/dict.mojo

./bazelw run //KGEN:mojo -- \
  debug KGEN/test/mojo-debug/Inputs/dict.mojo
```

The first Bazel invocation bootstraps Bazelisk and may download dependencies.
The `//KGEN:mojo` target includes `mojo-lldb`, `libMojoLLDB`, the standard
library, and related runtime files.

Afterward, optionally run the debugger regression suite:

```bash
./bazelw test //KGEN/test/mojo-debug:test
```

The repository's complete user-facing reference is
`mojo/docs/tools/debugging.mdx`.

## Debug the Mojo compiler implementation itself

If the objective is to debug the C++ implementation of the Mojo compiler,
rather than a Mojo program, use the checked-in Bazel debugger wrapper:

```bash
bazel/internal/bazel-debug.sh //KGEN/tools/mojo -- \
  run KGEN/test/mojo-debug/Inputs/dict.mojo
```
