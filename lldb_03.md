# Making Mojo LLDB expressions see stack variables

## Diagnosis

`expression --language mojo -- print(1 + 2)` works because it does not need
debugger state. `print(lhs + rhs)` fails during parsing because
`MojoExpressionParser::parse()` currently passes only persistent REPL variables
to `parseREPLExpression()`. The selected LLDB frame is never queried, so the
generated expression function has no parameters named `lhs` or `rhs`.

There are two halves to exposing a stack variable:

1. Add its name and Mojo type to the `variables` array passed to
   `parseREPLExpression()`. This makes name resolution and type checking work.
2. Add the same `lldb::VariableSP` to the expression `Materializer`, in exactly
   the same order. This makes LLDB put the variable's address into the context
   struct consumed by the compiled expression and write changes back afterward.

Changing only the parser side would compile an expression with a context layout
that does not match the materializer. Changing only the materializer side would
leave `lhs` and `rhs` unknown to the Mojo parser.

## Initial implementation

The first implementation is deliberately scoped to variables whose debug-info
type can be reconstructed in the expression parser's Mojo type system:

1. Construct `MojoExpressionParser` with the selected frame as its execution
   scope (falling back to the process if there is no frame), and save that
   `StackFrame`. Passing only the process loses the selected-frame identity.
2. In `parse()`, call
   `StackFrame::GetInScopeVariableList(false, false, true)` to collect arguments
   and locals with valid locations at the current PC.
3. Ignore artificial, unnamed, duplicate, and non-Mojo variables.
4. Print each debug-info MLIR type to text, then parse that text in the target's
   scratch `MojoTypeSystem`. This round trip is required because MLIR types from
   the module debug-info type system cannot safely be inserted into the scratch
   parser's different `MLIRContext`.
5. Recover source-level `Int`, `UInt`, and `Bool` for the corresponding scalar
   ABI types while generating the REPL aliases. Raw `!kgen.scalar` types do not
   have source-level methods such as `__add__`.
6. Put frame variables before persistent REPL variables. If names collide, the
   in-scope frame variable wins.
7. In `prepareForExecution()`, call `Materializer::AddVariable()` for those
   exact frame variables before registering persistent variables, preserving
   the parser/materializer field order.
8. Tell the REPL wrapper how many leading inputs are program-frame variables.
   LLDB materializes those as `Pointer[T]`; persistent REPL variables retain
   their existing `Pointer[Pointer[T]]` representation. Emitting two
   dereferences for a frame scalar treats its numeric value as an address (for
   example, `rhs == 22` faults at address `0x16`).
9. Replace Mojo's `_ = expression` unused-value fix-it with a generated
   `var __lldb_expr_result = expression`. Give that variable a direct
   `Pointer[T]` field backed by `Materializer::AddResultVariable()` and the
   `MojoUserExpression` result delegate. This publishes the value as `$R0`,
   `$R1`, and so on after dematerialization.
10. Compile the complete debugger-expression pipeline with
    `optimizationLevel = 0`. The expression parser owns a copied set of
    compilation options so both high-level KGEN compilation and final object
    compilation use O0.

The trace in `/tmp/mojo.txt` should then change from:

```text
input variables: 0
```

to output resembling:

```text
frame variables: 2
  lhs: !kgen.scalar<index>
  rhs: !kgen.scalar<index>
input variables: 2
```

## Build and validate

From `/home/kishmakov/Repos/modular`:

```bash
./bazelw build //KGEN:MojoLLDB --config=build-mojo
```

Then run the existing session:

```bash
cd /home/kishmakov/Templates
./run.sh
```

At the breakpoint, validate a bare result first:

```lldb
frame variable lhs rhs
expression --language mojo -- lhs + rhs
expression --language mojo -- 1 + 2
```

Expected output:

```text
(Int) $R0 = 42
(Int) $R1 = 3
```

LLDB also reports the generated result-variable fix-it after each value. Then
validate write-back separately:

```lldb
expression --language mojo -- answer = 100
frame variable answer
```

In the current `simple` binary, DWARF represents `lhs` and `rhs` with
`DW_AT_const_value`, so LLDB can read them but cannot write changes back to
them. `answer` has a writable stack location; the expected result of the test
above is `answer = 100`.

Direct LLDB results and REPL persistent variables both live in the persistent
expression state, but have different representations. Results have ordinary
Mojo types and are skipped as inputs to later expressions; REPL variables use
`REPLResultRefType`. Treating every entry as a REPL reference causes an
assertion on the second expression.

Useful diagnostics are:

```lldb
log enable lldb expr -v -f /tmp/mojo-lldb.log
```

and the existing raw trace at `/tmp/mojo.txt`.

## Expected follow-up work

The textual type transfer is enough for builtin scalar types such as `Int`, but
nominal structs and types involving declarations may need a real type importer:

- import the referenced Mojo declarations into the scratch parser context;
- recursively map debug-info types instead of only reparsing their spelling;
- define and test shadowing rules for nested lexical scopes;
- add tests for arguments, locals, optimized-out variables, mutation/write-back,
  structs, references, and name collisions with persistent REPL variables;
- share the frame-variable collection with code completion so completion and
  evaluation expose the same names.

Unsupported types should remain visible in verbose diagnostics and be skipped,
rather than making unrelated scalar expressions unusable. At the time of this
prototype, `Int`, `UInt`, and `Bool` scalar storage types are recovered; other
source types still require the general type importer described above.
