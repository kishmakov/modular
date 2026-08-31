#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$workspace_root"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to generate compile_commands.json" >&2
  exit 1
fi

exec_root="$(./bazelw info execution_root)"
compile_commands="$workspace_root/compile_commands.json"
compile_commands_tmp="$(mktemp "$workspace_root/.compile_commands.json.XXXXXX")"
trap 'rm -f "$compile_commands_tmp"' EXIT

./bazelw aquery \
  --config=build-mojo \
  --include_artifacts=false \
  --output=jsonproto \
  'mnemonic("CppCompile", //KGEN:MojoLLDB)' \
  | jq \
      --arg directory "$exec_root" \
      --arg workspace "$workspace_root" \
      '[
        .actions[]
        | .arguments as $argv
        | ($argv | index("-c")) as $source_flag
        | select($source_flag != null)
        | {
            directory: $directory,
            arguments: $argv,
            file: ($workspace + "/" + $argv[$source_flag + 1])
          }
      ]' \
      >"$compile_commands_tmp"

mv "$compile_commands_tmp" "$compile_commands"
trap - EXIT

echo "Wrote $compile_commands ($(jq length "$compile_commands") entries)"
