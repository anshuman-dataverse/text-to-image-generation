#!/usr/bin/env bash
# Build Milestone1_Report.pdf and Milestone1_Proposal.pdf (one page) from the Markdown.
# Needs pandoc and Google Chrome (headless print-to-PDF).
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for doc in Milestone1_Report:report.css Milestone1_Proposal:proposal.css; do
  name="${doc%%:*}"; css="${doc##*:}"
  pandoc "$name.md" -s --metadata title=" " -c "$css" --embed-resources -o "$tmp/$name.html"
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/$name.pdf" "file://$tmp/$name.html" 2>/dev/null
  echo "wrote $name.pdf"
done
