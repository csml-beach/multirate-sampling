#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PAPER_DIR="$REPO_ROOT/paper"
OUT_ROOT="$PAPER_DIR/submission"
OUT_DIR="$OUT_ROOT/arxiv"
TARBALL="$OUT_ROOT/arxiv-submission.tar.gz"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk not found on PATH" >&2
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Copy manuscript sources.
find "$PAPER_DIR" -maxdepth 1 -type f \( -name '*.tex' -o -name '*.bbl' -o -name '*.cls' -o -name '*.sty' \) \
  -exec cp {} "$OUT_DIR"/ \;

if [[ -d "$PAPER_DIR/Bib" ]]; then
  cp -R "$PAPER_DIR/Bib" "$OUT_DIR/Bib"
fi

# Collect and copy all figure paths referenced from the manuscript source.
python - "$PAPER_DIR" <<'PY' | while IFS= read -r rel; do
from pathlib import Path
import re
import sys

paper_dir = Path(sys.argv[1])
seen = set()
for tex_path in sorted(paper_dir.glob("*.tex")):
    text = tex_path.read_text()
    for match in re.findall(r"\.\./figures/[^}\s]+", text):
        if match not in seen:
            seen.add(match)
            print(match)
PY
  src="$REPO_ROOT/${rel#../}"
  dest="$OUT_DIR/${rel#../}"
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
done

# Rewrite the exported copy for arXiv:
# - blind submission must be false
# - review mode / line numbers must be disabled
# - figures must be self-contained within the bundle
python - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
for tex_path in out_dir.glob("*.tex"):
    text = tex_path.read_text()
    text = text.replace(r"\documentclass[preprint,12pt,review]{elsarticle}",
                        r"\documentclass[preprint,12pt]{elsarticle}")
    text = text.replace(r"\blindsubmissiontrue", r"\blindsubmissionfalse")
    text = text.replace(r"\linenumbers", "% \\linenumbers")
    text = text.replace("../figures/", "figures/")
    tex_path.write_text(text)
PY

# Verify the export compiles on its own.
(
  cd "$OUT_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)

# Build the upload archive from the exported bundle contents.
rm -f "$TARBALL"
(
  cd "$OUT_DIR"
  tar -czf "$TARBALL" .
)

echo "Created arXiv bundle:"
echo "  source dir: $OUT_DIR"
echo "  tarball:    $TARBALL"
