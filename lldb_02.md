### 1. lhs + rhs never reaches JIT execution

  I reproduced these commands:

  frame variable lhs rhs

  LLDB sees both values correctly:

  lhs = 20
  rhs = 22

  But:

  expression --language mojo -- print(lhs + rhs)

  returns silently, while:

  expression --language mojo -- print(1 + 2)

  successfully prints 3.

  The cause is that the Mojo expression parser only collects variables previously persisted by the Mojo REPL:

  state.collectPersistentVariables(variables);

  See KGEN/lib/MojoLLDB/ExpressionParser/MojoExpressionParser.cpp:354.

  It never collects local variables from the selected LLDB StackFrame. Consequently, the generated expression wrapper has no lhs or rhs arguments. The wrapper’s context structure is
  constructed exclusively from that variables collection in KGEN/lib/MojoTooling/ParserDriverREPL.cpp:429.

  So:

  LLDB can display lhs/rhs
          ↓
  Mojo expression parser does not import them
          ↓
  lhs/rhs are unresolved during Mojo parsing
          ↓
  No JIT execution

  ### 2. The parser error is swallowed

  MojoUserExpression::Parse() broadcasts diagnostics and then clears the DiagnosticManager:

  impl->expressionLogger.broadcastDiagnostics(diagnosticManager);
  diagnosticManager.Clear();

  See KGEN/lib/MojoLLDB/ExpressionParser/MojoUserExpression.cpp:357.

  However, MojoExpressionLogger::debugLog() only broadcasts an event; it does not write directly to the enabled LLDB log. The event reaches the LLDB expression log only if someone calls
  handleEvent().

  The Mojo REPL installs such a listener in KGEN/lib/MojoLLDB/REPL/MojoREPL.cpp:149, but the ordinary LLDB expression command does not appear to install one. Therefore the diagnostic is
  broadcast, nobody consumes it, and the diagnostic manager is cleared.

  That explains why your log stops at:

  == [UserExpression::Evaluate] Parsing expression lhs + rhs ==

  and why the console shows neither a result nor an unknown-name error.

  ### 3. Even constant expressions are not returned to LLDB

  I also tested:

  expression --language mojo -- 1 + 2

  Mojo applies this Fix-It:

  _ = 1 + 2

  The expression then compiles and executes. The LLDB trace ends with:

  Execution completed normally with no result

  The generated JIT IR contains only the internal __mojo_repl_expr_failed variable—not an integer result.

  The relevant code has a ResultDelegate and returns it from GetResultAfterDematerialization(), but nothing in the Mojo expression parser adds a result variable to the Materializer. The only
  materialized entities are persistent variables in KGEN/lib/MojoLLDB/ExpressionParser/MojoExpressionParser.cpp:564.

  Additionally, ordinary expression uses:

  options.replEnabled: false

  and shouldPersistVariable() rejects non-hidden variables outside REPL mode at KGEN/lib/MojoLLDB/ExpressionParser/MojoExpressionParser.cpp:287.

  ### Best next diagnostics

  Add direct logging—not MojoExpressionLogger events—at these locations:

  1. In LLDBMojoREPLListener::notifyDiagnostics():

     for (const auto &diag : diagnostics)
       traceOS << formatSMDiagnostic(diag, false) << "\n";

     This should expose the unresolved lhs/rhs diagnostic.

  2. Immediately after collectPersistentVariables():

     traceOS << "input variables: " << variables.size() << "\n";

     For this expression it should be zero; neither frame local is imported.

  3. After processVariablesForPersistence() or before prepareForExecution():

     traceOS << "new persistent variables: "
             << impl->newPersistentVariables.size() << "\n";

  4. In GetResultAfterDematerialization():

     traceOS << "result variable: "
             << static_cast<bool>(impl->resultDelegate.GetVariable()) << "\n";

     For 1 + 2, it should be null.

  The logging modification in MojoUserExpression::Parse() does not appear to cause the problem. It successfully records entry into parsing; it simply does not instrument the places where
  parsing fails or where the result should be created.
