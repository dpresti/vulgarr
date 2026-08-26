#!/usr/bin/env bash
# Installs this repo's git hooks (currently just pre-commit, see
# scripts/hooks/pre-commit) into .git/hooks/ -- run this once after cloning.
# .git/hooks/ isn't part of the repo itself, so hooks don't survive a clone
# on their own.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"

cp "$repo_root/scripts/hooks/pre-commit" "$repo_root/.git/hooks/pre-commit"
chmod +x "$repo_root/.git/hooks/pre-commit"
echo "Installed pre-commit hook."

if [ ! -f "$repo_root/.git-hooks-patterns.local" ]; then
  echo "No .git-hooks-patterns.local found -- the hook is installed but has nothing to check yet."
  echo "Copy .git-hooks-patterns.local.example to .git-hooks-patterns.local and fill in your own values if you want it active."
fi
