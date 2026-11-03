#!/usr/bin/env bash
# Build the milestone PDFs from the Markdown in docs/<milestone>/.
#   docs/build/build_pdfs.sh            # Milestone 1
#   docs/build/build_pdfs.sh milestone1
# Needs pandoc, Google Chrome (headless print-to-PDF) and pdfunite (poppler).
set -euo pipefail
BUILD="$(cd "$(dirname "$0")" && pwd)"
MS="${1:-milestone1}"
N="${MS#milestone}"
cd "$BUILD/../$MS"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for doc in "Milestone${N}_Report:report.css" "Milestone${N}_Proposal:proposal.css"; do
  name="${doc%%:*}"; css="$BUILD/${doc##*:}"
  [ -f "$name.md" ] || continue
  pandoc "$name.md" -s --metadata title=" " -c "$css" --lua-filter "$BUILD/github_links.lua" \
    --metadata docdir="docs/$MS/" --embed-resources -o "$tmp/$name.html"
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/$name.pdf" "file://$tmp/$name.html" 2>/dev/null
  echo "wrote docs/$MS/$name.pdf"
done

# One file for the Canvas upload: the one-page proposal followed by the full report
if [ -f "Milestone${N}_Proposal.pdf" ] && [ -f "Milestone${N}_Report.pdf" ]; then
  pdfunite "Milestone${N}_Proposal.pdf" "Milestone${N}_Report.pdf" "Milestone${N}_Submission.pdf"
  echo "wrote docs/$MS/Milestone${N}_Submission.pdf"
fi
