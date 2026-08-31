//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "MojoExpressionParser.h"
#include "../Logging/MojoExpressionLogger.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "../Utils/Binary.h"
#include "JITExecutionUnit.h"
#include "Logging.h"
#include "MojoDiagnostic.h"
#include "MojoExpressionVariable.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/Constants.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/SlicingUtils.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/ValueObject/ValueObject.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

static constexpr StringLiteral kLLDBResultVariableName = "__lldb_expr_result";

//===----------------------------------------------------------------------===//
// MojoExpressionParser::Impl
//===----------------------------------------------------------------------===//

struct MojoExpressionParser::Impl {
  Impl(ExecutionContextScope *exeScope, MojoUserExpression &expr,
       const EvaluateExpressionOptions &options);

  /// The expression being parsed.
  MojoUserExpression &expr;

  /// The type system associated with the evaluation of the current expression.
  MojoTypeSystem *typeSystem = nullptr;

  /// The compilation options to use when compiling.
  std::optional<KGEN::CompilationOptions> expressionCompilationOptions;
  const KGEN::CompilationOptions *compilationOptions = nullptr;

  /// The ObjectCompiler instance to use when parsing.
  std::unique_ptr<KGEN::ObjectCompiler> objCompiler;

  /// The parsed Mojo module.
  OwningOpRef<ModuleOp> mlirModule;

  /// The compiled object.
  OwningBinary<llvm::object::Binary> object;

  /// The options to use when evaluating the expression.
  EvaluateExpressionOptions options;

  /// A set of new persistent variables to be added to the persistent expression
  /// state if compilation of the expression succeeds.
  SmallVector<std::pair<StringRef, mlir::Type>> newPersistentVariables;

  /// The result variable produced for a bare expression, if any.
  std::optional<std::pair<StringRef, mlir::Type>> resultVariable;

  /// The frame selected when this expression parser was created.
  lldb::StackFrameWP frame;

  /// In-scope program variables passed to the expression materializer. Their
  /// order must match their order in the parser's input-variable list.
  SmallVector<lldb::VariableSP> frameVariables;

  /// Names supplied by frame variables. A frame variable shadows a persistent
  /// REPL variable with the same name.
  DenseSet<ConstString> frameVariableNames;

  /// The target on which expressions will be evaluated.
  lldb::TargetSP target;

  /// The expression logger for the current target.
  MojoExpressionLogger *expressionLogger;
};

MojoExpressionParser::Impl::Impl(ExecutionContextScope *exeScope,
                                 MojoUserExpression &expr,
                                 const EvaluateExpressionOptions &options)
    : expr(expr), options(options) {
  // Bail out if we don't have a valid execution context.
  frame = exeScope ? exeScope->CalculateStackFrame() : nullptr;
  target = exeScope ? exeScope->CalculateTarget() : nullptr;
  if (!target)
    return;

  expressionLogger = &MojoExpressionLogger::getLoggerForTarget(*target);

  // Grab the type system from the target, bailing out if we can't.
  auto typeSystemOr =
      target->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo);
  if (!typeSystemOr) {
    llvm::consumeError(typeSystemOr.takeError());
    return;
  }
  typeSystem = llvm::cast<MojoTypeSystem>(typeSystemOr.get().get());
  expressionCompilationOptions.emplace(
      typeSystem->getParserContext().getCompilationOptions());
  compilationOptions = &*expressionCompilationOptions;
  MLIRContext *ctx = typeSystem->getMLIRContext();

  // TODO(#33931) HACK, HACK, HACK!!!
  // To make CompilationOptions being properly passed to KGEN compiler
  // without breaking existing tests.
  // TODO(MOTO-247) workaround to LLVM Module splitting which works
  // for ORC JIT but not always for MCJIT.
  // Disable splitting and parallelize LLC pipeline
  // (which is based on splitting) for REPL.
  expressionCompilationOptions->enableLLVMPerFunctionSplitting = false;
  expressionCompilationOptions->enableParallelLLC = false;

  // Debugger expressions must preserve the stores and temporary values needed
  // for result materialization even when they otherwise appear dead to the
  // optimizer. REPL cells have no materialized result, so they keep the
  // target's optimization level to avoid a large performance regression.
  if (!options.GetREPLEnabled())
    expressionCompilationOptions->optimizationLevel = 0;

  PassManagerConfigOptions pmOptions;
  pmOptions.operationName = ModuleOp::getOperationName();

  // Create the compiler instance.
  auto compilerOr =
      ObjectCompiler::create(kMojoCacheBaseDirName, *compilationOptions,
                             /*isJIT=*/true, *ctx, pmOptions);

  if (failed(compilerOr))
    return;

  objCompiler = std::move(*compilerOr);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

namespace {
/// This class defines a simple raw ostream that can be used to emit colors when
/// processing diagnostic messages.
struct DiagnosticStream : public llvm::raw_string_ostream {
  DiagnosticStream(std::string &msg, bool supportsColors)
      : llvm::raw_string_ostream(msg) {
    enable_colors(supportsColors);
  }

  bool is_displayed() const override { return colors_enabled(); }
  bool has_colors() const override { return colors_enabled(); }
};
} // namespace

/// Format the given diagnostic into a string.
static std::string formatSMDiagnostic(const llvm::SMDiagnostic &diag,
                                      bool showColors) {
  std::string msg;
  DiagnosticStream msgOS(msg, showColors);

  // Set the default colors for the diagnostic printing. This ensures we use the
  // correct corresponding color for the diagnostic type.
  llvm::HighlightColor color(llvm::HighlightColor::Error);
  switch (diag.getKind()) {
  case llvm::SourceMgr::DK_Error:
    color = llvm::HighlightColor::Error;
    break;
  case llvm::SourceMgr::DK_Warning:
    color = llvm::HighlightColor::Warning;
    break;
  case llvm::SourceMgr::DK_Note:
    color = llvm::HighlightColor::Note;
    break;
  case llvm::SourceMgr::DK_Remark:
    color = llvm::HighlightColor::Remark;
    break;
  }
  llvm::WithColor colorOS(msgOS, color);

  diag.print("", msgOS, showColors, /*ShowKindLabel=*/false);
  return msg;
}

/// Return true if `name` collides with an identifier generated by the REPL
/// expression wrapper or by this plugin.
static bool isReservedExpressionName(StringRef name) {
  return name.starts_with("__lldb") || name.starts_with("__mojo_repl");
}

namespace {
/// Collect references that resolve to debugger-frame arguments in a structural
/// discovery parse. Generated wrapper references are ignored through location
/// mapping, as are references that resolve to declarations in the expression.
class FrameVariableReferenceCollector final : public MojoParserREPLListener,
                                              public LIT::ParserListener {
public:
  FrameVariableReferenceCollector(
      MojoParserContext &parserContext,
      const DenseSet<ConstString> &candidateFrameVariableNames)
      : parserContext(parserContext),
        candidateFrameVariableNames(candidateFrameVariableNames) {}

  void notifyWrappedExpr(StringRef) override {}
  void notifyFixedExpr(StringRef) override {}
  void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic>) override {}
  bool shouldPersistVariable(StringRef, mlir::Type) override { return false; }

  void onArgumentDecl(LIT::ASTDecl *decl, StringRef name,
                      llvm::SMLoc identifierLoc) override {
    // Only remember arguments synthesized by the wrapper. Arguments declared
    // by the user's expression have locations that map back into its buffer.
    if (parserContext.getREPLLocMapper().mapLocation(identifierLoc).isValid())
      return;
    if (candidateFrameVariableNames.contains(ConstString(name)))
      frameArgumentDecls.insert(decl);
  }

  void onRef(ArrayRef<LIT::ASTDecl *> decls, StringRef spelling,
             llvm::SMRange range) override {
    // References in the generated wrapper are implementation details. Only
    // references mapped back to source text participate in free-name
    // discovery.
    if (!parserContext.getREPLLocMapper().mapRange(range).isValid())
      return;
    if (llvm::any_of(decls, [&](LIT::ASTDecl *decl) {
          return frameArgumentDecls.contains(decl);
        }))
      referencedFrameVariableNames.insert(ConstString(spelling));
  }

  const DenseSet<ConstString> &getReferencedFrameVariableNames() const {
    return referencedFrameVariableNames;
  }

private:
  MojoParserContext &parserContext;
  const DenseSet<ConstString> &candidateFrameVariableNames;
  DenseSet<LIT::ASTDecl *> frameArgumentDecls;
  DenseSet<ConstString> referencedFrameVariableNames;
};

//===----------------------------------------------------------------------===//
// LLDBMojoREPLListener
//===----------------------------------------------------------------------===//

/// This class implements a parser listener that communicates between the Mojo
/// parser and the repl.
class LLDBMojoREPLListener : public MojoParserREPLListener {
  enum class FixItDisposition {
    /// Pass the parser's fixed text to LLDB unchanged.
    Passthrough,
    /// The only diagnostic binds an otherwise-unused value for LLDB.
    InternalResultRewrite,
    /// The result binding is combined with a user-visible correction.
    VisibleResultRewrite,
  };

public:
  LLDBMojoREPLListener(
      StringRef currentModuleName, StringRef exprText, MojoUserExpression &expr,
      DiagnosticManager &diagnosticManager,
      const EvaluateExpressionOptions &options,
      SmallVectorImpl<std::pair<StringRef, mlir::Type>> &newPersistentVariables,
      std::optional<std::pair<StringRef, mlir::Type>> &resultVariable,
      MojoExpressionLogger &expressionLogger)
      : currentModuleName(currentModuleName), exprText(exprText), expr(expr),
        diagnosticManager(diagnosticManager), options(options),
        newPersistentVariables(newPersistentVariables),
        resultVariable(resultVariable), expressionLogger(expressionLogger) {}
  ~LLDBMojoREPLListener() override = default;

  //===--------------------------------------------------------------------===//
  // Notifications

  void notifyWrappedExpr(StringRef wrappedExpr) override {
    expressionLogger.debugLog("Parsing the following code:\n{0}",
                              wrappedExpr.data());
  }

  void notifyFixedExpr(StringRef fixedExpr) override {
    if (fixItDisposition == FixItDisposition::Passthrough) {
      expr.setFixedText(fixedExpr);
      return;
    }

    // A bare Mojo expression is normally fixed to `_ = expression`, which
    // explicitly discards its value. For an LLDB expression, replace that
    // insertion with a generated result variable so the value can be
    // materialized and displayed by LLDB.
    if (discardFixItOffset && *discardFixItOffset <= fixedExpr.size() &&
        fixedExpr.drop_front(*discardFixItOffset).starts_with(kDiscardText)) {
      std::string resultExpr = fixedExpr.str();
      resultExpr.replace(*discardFixItOffset, kDiscardText.size(),
                         ("var " + kLLDBResultVariableName + " = ").str());

      if (fixItDisposition == FixItDisposition::InternalResultRewrite) {
        internalRewrite = std::move(resultExpr);
        return;
      }
      expr.setFixedText(resultExpr);
      return;
    }
    expr.setFixedText(fixedExpr);
  }

  //===--------------------------------------------------------------------===//
  // Internal rewrite

  /// Whether the parser's fix-it was solely our own result-variable rewrite,
  /// which `parse` re-parses itself rather than reporting to LLDB.
  bool hasInternalRewrite() const { return internalRewrite.has_value(); }

  /// Take the rewritten expression text. Only valid once.
  std::string takeInternalRewrite() { return std::move(*internalRewrite); }

  void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic> diagnostics) override {
    expressionLogger.debugLog("Found {0} diagnostic{1}\n", diagnostics.size(),
                              diagnostics.size() == 1 ? "" : "s");

    classifyFixIts(diagnostics);

    for (const llvm::SMDiagnostic &diag : diagnostics) {
      expressionLogger.debugLog("Diagnostic with fixits: {0}, message:\n{1}",
                                diag.getFixIts().size(), diag.getMessage());

      // If this is a warning or remark from a previous module, ignore it. This
      // removes problems with emitting multiple diagnostics for the same
      // expression.
      llvm::SourceMgr::DiagKind diagKind = diag.getKind();

      if (diagKind == llvm::SourceMgr::DK_Warning ||
          diagKind == llvm::SourceMgr::DK_Remark) {
        if (MojoPersistentExpressionState::isExpressionModuleName(
                diag.getFilename()) &&
            !diag.getFilename().ends_with(currentModuleName)) {
          lastDiagnosticIgnored = true;
          continue;
        }
      }

      // If this is a note and the previous diagnostic was ignored, ignore this
      // as well.
      if (diagKind == llvm::SourceMgr::DK_Note && lastDiagnosticIgnored)
        continue;
      lastDiagnosticIgnored = false;

      // Turn the diagnostic severity into LLDB's severity.
      lldb::Severity severity;
      switch (diagKind) {
      case llvm::SourceMgr::DK_Error:
        severity = lldb::eSeverityError;
        break;
      case llvm::SourceMgr::DK_Warning:
        severity = lldb::eSeverityWarning;
        break;
      case llvm::SourceMgr::DK_Remark:
        LLVM_FALLTHROUGH;
      case llvm::SourceMgr::DK_Note:
        severity = lldb::eSeverityInfo;
        break;
      }

      std::string msg = formatSMDiagnostic(diag, options.GetColorizeErrors());
      diagnosticManager.AddDiagnostic(std::make_unique<MojoDiagnostic>(
          msg, severity, !diag.getFixIts().empty()));
    }
  }

  //===--------------------------------------------------------------------===//
  // Queries

  bool shouldPersistVariable(StringRef name, mlir::Type type) override {
    if (isResultVariable(name)) {
      resultVariable.emplace(name, type);
      return true;
    }

    auto canPersist = [&] {
      // We always persist internal repl variables used for execution state.
      if (MojoParserContext::isHiddenPersistentVariable(name))
        return true;
      // Check if we were requested not to persist anything.
      if (options.GetSuppressPersistentResult())
        return false;
      // Only consider variables that were written by users, not those
      // generated by LLDB, which start with __lldb.
      if (name.starts_with("__lldb"))
        return false;
      // TODO: For now, we only persist variables in REPL mode. We should
      // define a policy for non-REPL mode (e.g. clang/swift using leading
      // $ for variable names to indicate persistence).
      if (!options.GetREPLEnabled())
        return false;
      return true;
    };

    if (canPersist()) {
      newPersistentVariables.emplace_back(name, type);
      return true;
    }
    return false;
  }

  bool isResultVariable(StringRef name) override {
    // A result is only produced when persistence was not suppressed, so a
    // variable that merely shares the name is not claimed as the result.
    return name == kLLDBResultVariableName &&
           !options.GetSuppressPersistentResult();
  }

private:
  StringRef currentModuleName;
  /// The source buffer holding the expression the parser was given. Fix-it
  /// ranges point into this buffer, which is a copy of `expr.Text()`.
  StringRef exprText;
  MojoUserExpression &expr;
  DiagnosticManager &diagnosticManager;
  const EvaluateExpressionOptions &options;
  SmallVectorImpl<std::pair<StringRef, mlir::Type>> &newPersistentVariables;
  std::optional<std::pair<StringRef, mlir::Type>> &resultVariable;
  MojoExpressionLogger &expressionLogger;

  /// Return the offsets of `range` within the expression text, if both ends
  /// point into that text.
  std::optional<std::pair<size_t, size_t>>
  getExprOffsets(llvm::SMRange range) const {
    if (!range.isValid())
      return std::nullopt;
    const char *start = range.Start.getPointer();
    const char *end = range.End.getPointer();
    if (start < exprText.begin() || start > exprText.end() || end < start ||
        end > exprText.end())
      return std::nullopt;
    return std::pair(start - exprText.begin(), end - exprText.begin());
  }

  /// Return true if `diag` is the parser warning whose fix-it discards an
  /// otherwise-unused value.
  bool isUnusedValueDiagnostic(const llvm::SMDiagnostic &diag) const {
    return !options.GetREPLEnabled() &&
           !options.GetSuppressPersistentResult() &&
           diag.getKind() == llvm::SourceMgr::DK_Warning &&
           diag.getMessage().contains(
               "value is unused; assign to '_' to discard the result") &&
           diag.getFixIts().size() == 1 &&
           diag.getFixIts().front().getText() == kDiscardText;
  }

  /// Classify the parser's fix-its and locate the unused-value insertion in
  /// the fully fixed text. Fix-it ranges refer to the original expression, so
  /// account for the text inserted or removed by every preceding fix-it.
  void classifyFixIts(ArrayRef<llvm::SMDiagnostic> diagnostics) {
    fixItDisposition = FixItDisposition::Passthrough;
    discardFixItOffset.reset();

    size_t previousEnd = 0;
    size_t fixedPrefixSize = 0;
    size_t numFixIts = 0;
    bool offsetsAreTrackable = true;

    for (const llvm::SMDiagnostic &diag : diagnostics) {
      bool isUnusedValue = isUnusedValueDiagnostic(diag);
      for (const llvm::SMFixIt &fixIt : diag.getFixIts()) {
        ++numFixIts;
        std::optional<std::pair<size_t, size_t>> offsets =
            getExprOffsets(fixIt.getRange());
        if (!offsets || offsets->first < previousEnd) {
          offsetsAreTrackable = false;
          continue;
        }
        if (!offsetsAreTrackable)
          continue;

        fixedPrefixSize += offsets->first - previousEnd;
        if (isUnusedValue)
          discardFixItOffset = fixedPrefixSize;
        fixedPrefixSize += fixIt.getText().size();
        previousEnd = offsets->second;
      }
    }

    if (!discardFixItOffset)
      return;

    // Only the exact single-diagnostic case is internal. Any accompanying
    // diagnostic follows LLDB's normal, user-visible Fix-It path so it cannot
    // be hidden when the internal rewrite is re-parsed.
    bool isOnlyUnusedValueDiagnostic =
        diagnostics.size() == 1 && numFixIts == 1 &&
        isUnusedValueDiagnostic(diagnostics.front());
    fixItDisposition = isOnlyUnusedValueDiagnostic
                           ? FixItDisposition::InternalResultRewrite
                           : FixItDisposition::VisibleResultRewrite;
  }

  /// The text the parser inserts to discard an unused expression value.
  static constexpr StringLiteral kDiscardText = "_ = ";

  /// A flag indicating if that the last processed diagnostic was ignored.
  bool lastDiagnosticIgnored = false;

  /// Where the parser inserted the discard fix-it for a bare expression, if it
  /// did so.
  std::optional<size_t> discardFixItOffset;

  /// How the current batch of parser fix-its should be exposed to LLDB.
  FixItDisposition fixItDisposition = FixItDisposition::Passthrough;

  /// The rewritten expression text, when the only fix-it was our own.
  std::optional<std::string> internalRewrite;
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoExpressionParser
//===----------------------------------------------------------------------===//

MojoExpressionParser::MojoExpressionParser(
    ExecutionContextScope *exeScope, MojoUserExpression &expr,
    const EvaluateExpressionOptions &options)
    : impl(std::make_unique<Impl>(exeScope, expr, options)) {}

MojoExpressionParser::~MojoExpressionParser() = default;

M::LogicalResult
MojoExpressionParser::parse(MojoPersistentExpressionState &state,
                            DiagnosticManager &diagnosticManager) {
  if (!impl->objCompiler) {
    impl->expressionLogger->errorLog("No ObjectCompiler");
    return failure();
  }

  MojoParserContext &parserContext = impl->typeSystem->getParserContext();
  MLIRContext *ctx = impl->typeSystem->getMLIRContext();
  llvm::SourceMgr &sourceMgr = parserContext.getSourceMgr();

  // Register the source manager diagnostic handler so we get all the MLIR
  // diagnostics through the handler we already have and so it's all forwarded
  // to the LLDB streams. If the handler can't use the source manager for an
  // error, it'll print to errStream, which we will flush if it's non-empty on
  // scope exit.
  std::string errs;
  llvm::raw_string_ostream errStream(errs);
  mlir::SourceMgrDiagnosticHandler handler(sourceMgr, ctx, errStream);

  // On scope exit, if we've printed any errors make sure to log them.
  auto printOnError = llvm::scope_exit([&]() {
    if (errs.empty())
      return;
    impl->expressionLogger->errorLog("{0}", errs);
  });

  // Collect the frame variables that can be represented in the expression's
  // Mojo context. A structural discovery parse below determines which of these
  // candidates the expression actually references before any are materialized.
  struct FrameVariableCandidate {
    lldb::VariableSP variable;
    ConstString name;
    mlir::Type type;
  };
  SmallVector<FrameVariableCandidate> candidateFrameVariables;
  DenseSet<ConstString> candidateFrameVariableNames;
  impl->frameVariables.clear();
  impl->frameVariableNames.clear();
  DenseSet<ConstString> seenFrameVariableNames;
  if (lldb::StackFrameSP frame = impl->frame.lock()) {
    lldb::VariableListSP frameVariableList =
        frame->GetInScopeVariableList(/*getFileGlobals=*/false,
                                      /*includeSyntheticVars=*/false,
                                      /*mustHaveValidLocation=*/true);
    if (frameVariableList) {
      for (lldb::VariableSP variable : *frameVariableList) {
        if (!variable || variable->IsArtificial())
          continue;

        // Variable::GetName() and Variable::GetType() are not exported by
        // liblldb. Go through the public StackFrame/ValueObject boundary to
        // obtain the same information from a loadable language plugin.
        lldb::ValueObjectSP value = frame->GetValueObjectForFrameVariable(
            variable, lldb::eNoDynamicValues);
        if (!value)
          continue;

        ConstString name = value->GetName();
        if (name.IsEmpty())
          continue;

        // A name generated by the expression wrapper cannot be passed through
        // it, and one shadowing the generated result would break the result
        // field's layout.
        if (isReservedExpressionName(name.GetStringRef()))
          continue;

        // An inner scope wins over an outer scope of the same name; the list
        // is ordered innermost first.
        if (!seenFrameVariableNames.insert(name).second)
          continue;

        CompilerType compilerType = value->GetCompilerType();
        if (!compilerType.IsValid() ||
            !compilerType.GetTypeSystem<MojoTypeSystem>())
          continue;

        // Debug-info types live in a module-specific MLIRContext, while
        // expression parsing uses the target's scratch MojoTypeSystem. Reparse
        // the type spelling in the scratch context instead of passing an MLIR
        // type across contexts.
        mlir::Type debugMLIRType =
            mlir::Type::getFromOpaquePointer(compilerType.GetOpaqueQualType());
        std::string typeName = mlir::debugString(debugMLIRType);
        CompilerType expressionCompilerType =
            impl->typeSystem->getBuiltinTypeFromMLIRTypeName(typeName);
        if (!expressionCompilerType.IsValid()) {
          impl->expressionLogger->debugLog(
              "Skipping frame variable '{0}': cannot import Mojo type {1}",
              name.GetStringRef(), typeName);
          continue;
        }

        mlir::Type expressionType = mlir::Type::getFromOpaquePointer(
            expressionCompilerType.GetOpaqueQualType());
        candidateFrameVariables.push_back(
            {std::move(variable), name, expressionType});
        candidateFrameVariableNames.insert(name);
        // Note that `candidateFrameVariableNames` deliberately lags
        // `seenFrameVariableNames`: a frame variable whose type could not be
        // imported is not a usable input, so a persistent variable of the same
        // name stays visible to the expression.
      }
    }
  }

  // Parse once with every usable frame variable in scope. The parser listener
  // records only source references that resolve to the generated frame
  // arguments, naturally excluding comments, strings, member names, and names
  // declared by the expression itself. This pass never materializes values.
  DenseSet<ConstString> referencedFrameVariableNames;
  if (!candidateFrameVariables.empty()) {
    SmallVector<std::pair<StringRef, mlir::Type>> discoveryVariables;
    for (const FrameVariableCandidate &candidate : candidateFrameVariables)
      discoveryVariables.emplace_back(candidate.name.GetStringRef(),
                                      candidate.type);
    state.collectPersistentVariables(discoveryVariables,
                                     candidateFrameVariableNames);

    FrameVariableReferenceCollector referenceCollector(
        parserContext, candidateFrameVariableNames);
    auto [discoveryExpressionId, discoveryModuleName] =
        state.getNextExpressionModuleName();
    std::string discoveryFnName =
        ("__lldb_expr_discovery__" + Twine(discoveryExpressionId)).str();
    int discoveryFileId = sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(impl->expr.Text(),
                                             discoveryModuleName),
        llvm::SMLoc());
    {
      LIT::ParserListener *previousParserListener =
          parserContext.getSharedState().parserListener;
      parserContext.getSharedState().parserListener = &referenceCollector;
      auto restoreParserListener = llvm::scope_exit([&] {
        parserContext.getSharedState().parserListener = previousParserListener;
      });

      MojoParserContext::ParsedREPLExpr discoveryResult =
          parserContext.parseREPLExpression(
              referenceCollector, discoveryFileId, discoveryFnName,
              discoveryVariables,
              /*numProgramVariables=*/candidateFrameVariables.size());
      if (discoveryResult.isValid())
        parserContext.removeLastREPLExpression();
    }

    referencedFrameVariableNames.insert(
        referenceCollector.getReferencedFrameVariableNames().begin(),
        referenceCollector.getReferencedFrameVariableNames().end());
  }

  impl->expressionLogger->debugLog(
      "Discovery parse selected {0} of {1} frame variables\n",
      referencedFrameVariableNames.size(), candidateFrameVariables.size());

  SmallVector<std::pair<StringRef, mlir::Type>> variables;
  for (FrameVariableCandidate &candidate : candidateFrameVariables) {
    if (!referencedFrameVariableNames.contains(candidate.name))
      continue;
    variables.emplace_back(candidate.name.GetStringRef(), candidate.type);
    impl->frameVariables.push_back(std::move(candidate.variable));
    impl->frameVariableNames.insert(candidate.name);
  }

  // Persistent REPL variables follow frame variables in both the parser input
  // and the materializer. The selected frame takes precedence on collisions.
  state.collectPersistentVariables(variables, impl->frameVariableNames);

  // Parse the expression. `parseExpr` is used twice: once for the text the
  // user typed, and again if the only thing the parser fixed was our own
  // result-variable rewrite.
  std::optional<LLDBMojoREPLListener> listener;
  MojoParserContext::ParsedREPLExpr result;
  // Both outlive the lambda: `exprModuleName` backs a StringRef held by the
  // listener, and `exprFnName` names the compiled entry point further down.
  std::string exprModuleName;
  std::string exprFnName;
  auto parseExpr = [&](StringRef exprText) {
    size_t expressionId;
    std::tie(expressionId, exprModuleName) =
        state.getNextExpressionModuleName();
    // Create a function name for the expression. This string must be a valid
    // Mojo identifier.
    exprFnName = ("__lldb_expr__" + Twine(expressionId)).str();
    int exprFileId = sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(exprText, exprModuleName),
        llvm::SMLoc());
    listener.emplace(exprModuleName,
                     sourceMgr.getMemoryBuffer(exprFileId)->getBuffer(),
                     impl->expr, diagnosticManager, impl->options,
                     impl->newPersistentVariables, impl->resultVariable,
                     *impl->expressionLogger);
    result = parserContext.parseREPLExpression(
        *listener, exprFileId, exprFnName, variables,
        /*numProgramVariables=*/impl->frameVariables.size());
  };
  parseExpr(impl->expr.Text());

  // Binding a bare expression to a generated result variable is an internal
  // detail rather than a correction to what the user typed, so it is re-parsed
  // here instead of being handed back to LLDB as a fix-it. Going through
  // LLDB's fix-it channel would make it announce the rewritten text on every
  // bare expression.
  if (listener->hasInternalRewrite() && impl->options.GetAutoApplyFixIts()) {
    std::string rewritten = listener->takeInternalRewrite();
    impl->expressionLogger->debugLog(
        "Rewrote the input to capture its result:\n{0}", rewritten);

    // Drop everything the first parse produced: the un-fixed module must not
    // stay in the REPL history, its diagnostics describe text that is being
    // replaced, and the variables it collected would otherwise be counted
    // twice.
    if (result.isValid())
      parserContext.removeLastREPLExpression();
    diagnosticManager.Clear();
    impl->newPersistentVariables.clear();
    impl->resultVariable.reset();

    parseExpr(rewritten);
  }

  // Record the entry point only once the text is settled: `setFunctionName`
  // may be called a single time, and only `prepareForExecution` reads it back.
  impl->expr.setFunctionName(exprFnName);

  // If the parser supplied a fixed expression, abort processing and use that
  // expression instead.
  if (!impl->expr.GetFixedText().empty() &&
      impl->options.GetAutoApplyFixIts()) {
    impl->expressionLogger->debugLog(
        "Rewrote the input, next parse will be the fixed code:\n{0}",
        impl->expr.GetFixedText());

    // If we have a fixed expression string, we're going to fail here to let
    // LLDB retry execution with the fixed expression. Before then, we need to
    // emit all of the fixed diagnostics that were collected, given that these
    // won't be shown on the next parse.
    auto filterFn = [](MojoDiagnostic &diag) { return diag.hadFixits(); };
    impl->expressionLogger->broadcastDiagnostics(diagnosticManager, filterFn);
    diagnosticManager.Clear();

    // If the parser was actually successful, make sure to reset it so that we
    // don't include the un-fixed module in the REPL history.
    if (result.isValid())
      parserContext.removeLastREPLExpression();
    return failure();
  }

  if (!result.isValid())
    return failure();
  impl->expressionLogger->debugLog("Parsed module successfully");

  // Setup a diagnostic handler to process diagnostics emitted during lowering.
  struct MLIRDiagnosticHandlerContext {
    LLDBMojoREPLListener &listener;
    MojoParserContext &parserContext;
  };
  MLIRDiagnosticHandlerContext handlerContext{*listener, parserContext};
  sourceMgr.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *context) {
        auto *ctx = static_cast<MLIRDiagnosticHandlerContext *>(context);
        ctx->listener.notifyDiagnostics(
            ctx->parserContext.getREPLLocMapper().mapDiagnostic(diag));
      },
      &handlerContext);

  // Functor containing various cleanup performed in the case of an error.
  auto returnErrorCleanup = [&] {
    // If we encounter an error anywhere during compilation, make sure the
    // parser doesn't include this expression in the REPL history.
    parserContext.removeLastREPLExpression();
    return failure();
  };

  // Create a clone of the parser module so that we can compile it without
  // thrashing on the current parser state.
  auto exprFn = cast<LIT::FnOp>(result.exprFnDecl.getIfOperation());
  exprFn.setLinkageNameAttr(
      LinkageNameAttr::get(exprFn->getContext(), exprFnName));
  mlir::IRMapping mapping;
  OwningOpRef<ModuleOp> module =
      KGEN::LIT::cloneDeclModuleForCompilation(*result.moduleDecl, mapping);
#ifndef MODULAR_PRODUCTION
  if (failed(mlir::verify(*module)))
    return returnErrorCleanup();
#endif // MODULAR_PRODUCTION

  // Set the environment (defines) for the module.
  extendWithModularEnvAttr(*module, nullptr);

  // Ensure the expression function in the cloned module gets c-exported.
  auto clonedExprFn = cast<LIT::FnOp>(mapping.lookup(&*exprFn));
  clonedExprFn.setExported();
  // Set the C ABI effect
  auto sigGen = clonedExprFn.getFuncTypeGenerator();
  auto body = sigGen.getBody();
  auto newBody = body.getWithFnEffects(body.getFnEffects().setCABI(true));
  clonedExprFn.setFuncTypeGenerator(LIT::FnTypeGeneratorType::get(
      sigGen.getInputParamTypes(), newBody, sigGen.getParamListAttrs()));

  // Log the pre-elaboration module.
  Log *logChannel = GetLog(LLDBLog::Expressions);
  bool isVerboseLoggingEnabled = logChannel && logChannel->GetVerbose();

  std::string preElaborationModuleLog;
  llvm::raw_string_ostream preElaborationLogStream(preElaborationModuleLog);

  PassManagerConfigOptions pmOptions;
  if (isVerboseLoggingEnabled) {
    pmOptions.irPrintingOptions.enable = true;
    pmOptions.irPrintingOptions.passName = "ElaborateGenerators";
    pmOptions.irPrintingOptions.out = &preElaborationLogStream;
  }
  pmOptions.operationName = ModuleOp::getOperationName();

  KGEN::KGENCompiler kgenCompiler(*impl->typeSystem->getMLIRContext(),
                                  *impl->compilationOptions, pmOptions);

  //// Get the target info to use for compilation.
  TargetInfoAttr targetInfo = impl->typeSystem->GetTargetInfo();
  if (!targetInfo)
    return failure();

  // Run the elaboration pipeline.
  ErrorOrSuccess compilerResult = kgenCompiler.runKGENPipeline(
      *module, targetInfo, impl->objCompiler->getTransformCache().copy(),
      AsyncValueRef<Chain>::createReady(impl->typeSystem->getCPUDevice()));
  if (compilerResult.isError())
    return returnErrorCleanup();

  if (isVerboseLoggingEnabled) {
    impl->expressionLogger->dumpIR("Pre-elaboration module:\n{0}",
                                   preElaborationModuleLog);
    impl->expressionLogger->dumpIR("Elaborated module:\n{0}", *module);
  }

  // Compile the module to a standalone archive.
  SymbolTable symbolTable(*module);
  ExportMap exportedSymbols;
  exportedSymbols.insert({StringAttr::get(module->getContext(), exprFnName),
                          ExportKind::Exported});
  OwningOpRef<ModuleOp> sliceModule =
      produceStandaloneModule(symbolTable, exportedSymbols);
  auto bufferOr = impl->objCompiler->emitArchive(std::move(sliceModule));
  if (bufferOr.isError()) {
    impl->expressionLogger->errorLog(
        "Failed to produce standalone archive: {0}", bufferOr.getError());
    return returnErrorCleanup();
  }

  auto objectOr = toModularErrorOr(
      llvm::object::createBinary((*bufferOr)->getMemBufferRef()));

  if (objectOr.isError()) {
    impl->expressionLogger->errorLog("Failed to create the binary object: {0}",
                                     objectOr.getError());
    return returnErrorCleanup();
  }

  impl->mlirModule = std::move(module);
  impl->object = OwningBinary<llvm::object::Binary>(std::move(*objectOr),
                                                    std::move(*bufferOr));

  return success();
}

Status MojoExpressionParser::prepareForExecution(
    lldb::addr_t &funcAddr, lldb::addr_t &funcEnd,
    std::shared_ptr<JITExecutionUnit> &executionUnit, ExecutionContext &exeCtx,
    ExecutionPolicy executionPolicy, bool keepResultInMemory) {
  // Grab the module and standalone archive built during the parse phase.
  // NOTE: impl->mlirModule and impl->archive will be nullptr after this!
  // Luckily, expressions are generally destroyed shortly after this, so we
  // don't have to be too concerned - just something to be aware of.
  OwningOpRef<ModuleOp> mlirModule = std::move(impl->mlirModule);
  if (!mlirModule)
    return Status::FromErrorString("Can't prepare a NULL module for execution");

  // Retrieve an appropriate symbol context.
  SymbolContext sc;
  if (const lldb::StackFrameSP &frame = exeCtx.GetFrameSP())
    sc = frame->GetSymbolContext(lldb::eSymbolContextEverything);
  else if (const lldb::TargetSP &target = exeCtx.GetTargetSP())
    sc.target_sp = target;

  // Extract the target features.
  SmallVector<StringRef> splitFeatures;
  StringRef(impl->compilationOptions->targetFeatures).split(splitFeatures, ",");
  std::vector<std::string> features(splitFeatures.begin(), splitFeatures.end());

  // Build the IR execution unit responsible for executing the generated IR.
  ConstString functionName(impl->expr.FunctionName());
  SymbolTable symbolTable(*mlirModule);
  ExportMap exportedSymbols;
  exportedSymbols.insert(
      {StringAttr::get(mlirModule->getContext(), functionName.GetStringRef()),
       ExportKind::Exported});
  executionUnit = std::make_shared<JITExecutionUnit>(
      symbolTable, exportedSymbols, std::move(impl->object), functionName,
      exeCtx.GetTargetSP(), sc, features);

  // Extract the function information for the expression entry point.
  Status error = executionUnit->getRunnableInfo(funcAddr, funcEnd);
  if (error.Fail())
    return error;

  auto *persistentState = static_cast<MojoPersistentExpressionState *>(
      impl->typeSystem->GetPersistentExpressionState());

  // The input fields of the generated context struct must be materialized
  // whatever happens to the result, because the struct's layout was fixed when
  // the expression was compiled. Register frame variables first so that layout
  // matches the input-variable order used to compile the expression wrapper.
  for (lldb::VariableSP &variable : impl->frameVariables) {
    impl->expr.GetMaterializer()->AddVariable(variable, error);
    if (error.Fail())
      return error;
  }

  // Register the current persistent variables with the materializer. This must
  // walk the state exactly the way `parse` did when it built the parser's
  // input list.
  persistentState->forEachInputVariable(
      impl->frameVariableNames,
      [&](lldb::ExpressionVariableSP &var, mlir::Type) {
        if (error.Fail())
          return;
        impl->expr.GetMaterializer()->AddPersistentVariable(var, nullptr,
                                                            error);
      });
  if (error.Fail())
    return error;

  // Nothing produced by this evaluation outlives it, so there are no output
  // fields to register.
  if (!keepResultInMemory)
    return error;

  // Compute the target info to use for the persistent variable state.
  lldb_private::Process *process = exeCtx.GetProcessPtr();
  lldb::ByteOrder byteOrder = process->GetByteOrder();
  size_t addressByteSize = process->GetAddressByteSize();

  // The output fields of the generated context struct are emitted sorted by
  // name (see `processVariablesForPersistence`), and the result variable is
  // just another output field. Register the outputs in that same order so the
  // materializer's offsets line up with the compiled struct.
  SmallVector<std::pair<StringRef, mlir::Type>> outputs(
      impl->newPersistentVariables);
  if (impl->resultVariable)
    outputs.push_back(*impl->resultVariable);
  llvm::sort(outputs, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  // Register the newly created persistent variables and, if the expression
  // produced one, LLDB's result.
  std::vector<lldb::ExpressionVariableSP> persistentVariables;
  for (auto [name, mlirType] : outputs) {
    // The result of a bare expression is written straight into storage owned
    // by the materializer rather than into the persistent state.
    if (impl->resultVariable && name == impl->resultVariable->first) {
      CompilerType resultType(
          impl->typeSystem->weak_from_this(),
          const_cast<void *>(mlirType.getAsOpaquePointer()));
      impl->expr.addResultVariable(resultType, keepResultInMemory, error);
      if (error.Fail())
        return error;
      continue;
    }

    // All persistent variables in the REPL are references, so wrap them in a
    // reference type.
    auto ptr = LIT::REPLResultRefType::get(mlirType);
    CompilerType lldbType(impl->typeSystem->weak_from_this(),
                          const_cast<void *>(ptr.getAsOpaquePointer()));
    lldb::ExpressionVariableSP var = persistentState->CreatePersistentVariable(
        exeCtx.GetBestExecutionContextScope(), ConstString(name), lldbType,
        byteOrder, addressByteSize);
    if (!var) {
      error = Status::FromErrorString("failed to create persistent variable");
      return error;
    }

    // Mark the variable as persistent, and notify LLDB that it needs to be
    // allocated.
    var->m_frozen_sp->SetHasCompleteType();
    var->m_flags |= ExpressionVariable::EVKeepInTarget;
    var->m_flags |= ExpressionVariable::EVIsLLDBAllocated;
    var->m_flags |= ExpressionVariable::EVNeedsAllocation;

    // Adding the variable to the expression materializer.
    impl->expr.GetMaterializer()->AddPersistentVariable(var, nullptr, error);
    if (error.Fail())
      return error;
    persistentVariables.emplace_back(std::move(var));
  }

  // If a valid execution unit was produced and there is more than one external
  // function in the execution unit, it needs to keep living even if it's not
  // top level, because the result could refer to that function, register it if
  // necessary.
  //
  // In REPL mode, always persist the execution unit to keep JIT-section memory
  // alive. String literals constructed from `StringLiteral` store a raw pointer
  // into the JIT data section (via `pop.string.address`) without heap-copying,
  // so freeing the execution unit's JIT sections would leave persisted String
  // variables with dangling data pointers.
  std::shared_ptr<JITExecutionUnit> persistedExecutionUnit;
  if (executionUnit &&
      (impl->options.GetExecutionPolicy() == eExecutionPolicyTopLevel ||
       impl->options.GetREPLEnabled() ||
       executionUnit->getJittedFunctions().size() > 1)) {
    persistedExecutionUnit = executionUnit;
  }

  // Register the persisted state for this execution.
  persistentState->registerExpressionInstance(std::move(persistedExecutionUnit),
                                              std::move(persistentVariables),
                                              impl->expr.getPythonModuleName());
  return error;
}
