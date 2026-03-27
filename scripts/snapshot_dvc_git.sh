#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/snapshot_dvc_git.sh [options]

Options:
  -m, --message MSG   Commit message. Defaults to a timestamped snapshot message.
      --dry-run       Print actions without changing Git or DVC state.
      --push          Run git push after committing.
      --dvc-push      Run dvc push after updating DVC metadata.
  -h, --help          Show this help text.
EOF
}

log() {
  printf '%s\n' "$*"
}

run() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

stage_if_exists() {
  local existing=()
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      existing+=("$path")
    fi
  done
  if [[ ${#existing[@]} -gt 0 ]]; then
    run git add -u -- "${existing[@]}"
  fi
}

stage_safe_untracked() {
  local path
  while IFS= read -r path; do
    case "$path" in
      README.md|.gitignore|.dvcignore|docs/*.md|scripts/*.sh|scripts/*.py|scripts/*.md|jax/*.py|jax/*.md|jax/*.sh|jax/*.yml|jax/*.yaml|paper/*.tex|paper/*.md|paper/*.pdf|paper/Bib/*.bib|.github/*.yml|.github/*.yaml|.github/*.md)
        run git add -- "$path"
        ;;
    esac
  done < <(git ls-files --others --exclude-standard -- README.md .gitignore .dvcignore docs jax scripts paper .github 2>/dev/null || true)
}

DRY_RUN=0
DO_PUSH=0
DO_DVC_PUSH=0
COMMIT_MSG=""
ARTIFACT_DIRS=(figures metrics animations)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 1; }
      COMMIT_MSG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --push)
      DO_PUSH=1
      shift
      ;;
    --dvc-push)
      DO_DVC_PUSH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -x "/Users/arash/.pyenv/versions/jax/bin/dvc" ]]; then
  DVC_BIN="/Users/arash/.pyenv/versions/jax/bin/dvc"
elif command -v dvc >/dev/null 2>&1; then
  DVC_BIN="$(command -v dvc)"
else
  echo "dvc not found" >&2
  exit 1
fi

if [[ ! -d .dvc ]]; then
  run "$DVC_BIN" init -q
fi

for dir in "${ARTIFACT_DIRS[@]}"; do
  if [[ -d "$dir" ]] && find "$dir" -mindepth 1 -print -quit | grep -q .; then
    run "$DVC_BIN" add "$dir"
  fi
done

for dvc_file in figures.dvc metrics.dvc animations.dvc; do
  if [[ -e "$dvc_file" ]]; then
    run git add -- "$dvc_file"
  fi
done

stage_if_exists README.md docs jax scripts paper .github .gitignore .dvcignore
stage_safe_untracked

if git diff --cached --quiet; then
  log "No staged changes to commit."
  exit 0
fi

if [[ -z "$COMMIT_MSG" ]]; then
  COMMIT_MSG="snapshot: $(date '+%Y-%m-%d %H:%M:%S')"
fi

run git commit -m "$COMMIT_MSG"

if [[ "$DO_DVC_PUSH" -eq 1 ]]; then
  run "$DVC_BIN" push
fi

if [[ "$DO_PUSH" -eq 1 ]]; then
  run git push
fi
