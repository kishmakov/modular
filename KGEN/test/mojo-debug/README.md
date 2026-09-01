# Mojo Debug Tests

There are two main ways to test LLDB's ability to parse DWARF and other debug
info sections and provide a nice debugging experience: llvm-lit tests and SB API
tests.

## llvm-lit tests

These tests assert on the LLDB CLI's output. Prefer `frame variable` (aka `v`)
for inspecting a variable: it reads the value straight out of the process
without compiling anything, so it is faster and it isolates the DWARF and
formatter behaviour that these tests are usually about.

Use `expression` (aka `p`) when the JIT path itself is what is under test, as
in `expression.lldb`. The expression is parsed, compiled, and executed in the
target, and its inputs are the frame locals the expression names.

An advantage of `frame variable` is that it allows static dereferencing of
pointers with the `*` operator, e.g. `frame variable *a_pointer`.

Besides that, it also allows variable paths, e.g.
`frame variable a_struct.a_member.a_nested_member`.

Sadly, the output of `frame variable` can be very limiting and doesn't show
type names by default, nor lets you easily query specific information about
a type or a variable (e.g. is this int type signed or unsigned?).

## SB API tests

These tests are based on the public LLDB SB API and can be extremely precise.
I recommend looking for the following files as reference for how to query
variable and type information:

- third-party/llvm-project/lldb/include/lldb/API/SBFrame.h
- third-party/llvm-project/lldb/include/lldb/API/SBValue.h
- third-party/llvm-project/lldb/include/lldb/API/SBType.h

You can use `KGEN/unittests/mojo-debug/PrimitiveTypesTest.cpp` as an example of
how to write this kind of tests.

For further details, the list of local variables can be accessed through
`StopContext.frame.FindVariable()` or
`StopContext.frame.FindVariableGetValueForVariablePath()`, which resemble the
`frame variable` command. Once you have a variable, which is an `SBValue`
object, you can start querying its value if it's a primitive via the
`GetValue()` family of functions, its summary if it has a synthetic formatter
via `GetSummary()`, or access its fields via `GetChildAtIndex`. Similarly, the
type can be gotten via `GetType()`.

As a final tip, don't hardcode line numbers in your tests. They break often, so
it's better to add markers in the source code and do string search for them
to use them in asserts. The class `SourceFile` should help with that.
