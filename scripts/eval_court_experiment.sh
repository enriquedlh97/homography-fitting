#!/bin/bash
# Full evaluation for a court-floor experiment:
#   1. Run quality_eval.py (metrics + banner crops)
#   2. Extract court-floor crops at standard frames (wide + tight)
# Usage: ./scripts/eval_court_experiment.sh experiments/<dir> [original_video]

set -e

EXPERIMENT_DIR="${1:?Usage: $0 <experiment_dir> [original_video]}"
ORIGINAL_VIDEO="${2:-data/melbourne-walking-over-logo.mov}"

if [ ! -d "$EXPERIMENT_DIR" ]; then
  echo "Error: $EXPERIMENT_DIR is not a directory"
  exit 1
fi

COMPOSITED="$EXPERIMENT_DIR/outputs/composited.mp4"
if [ ! -f "$COMPOSITED" ]; then
  echo "Error: $COMPOSITED not found"
  exit 1
fi

# Step 1: quality metrics + banner crops
echo "============ Quality metrics ============"
UV_CACHE_DIR=.uv-cache uv run python scripts/quality_eval.py \
  --experiment "$EXPERIMENT_DIR" \
  --original "$ORIGINAL_VIDEO" 2>&1 | tail -20

# Step 2: court-floor crops at standard frames
COURT_CROPS_DIR="$EXPERIMENT_DIR/crops_court"
mkdir -p "$COURT_CROPS_DIR"

echo ""
echo "============ Court floor crops ============"
echo "Saving to $COURT_CROPS_DIR"

# Standard reference frames for the walkover clip
FRAMES=(0 50 100 200 300 500 700 730)

for f in "${FRAMES[@]}"; do
  # Wide: full court area around logo (1400x500 starting at 250,600)
  ffmpeg -y -i "$COMPOSITED" -vf "select=eq(n\,${f}),crop=1400:500:250:600" \
    -vframes 1 "$COURT_CROPS_DIR/wide_${f}.png" 2>/dev/null
  # Tight: foot region (400x200 starting at 700,850)
  ffmpeg -y -i "$COMPOSITED" -vf "select=eq(n\,${f}),crop=400:200:700:850" \
    -vframes 1 "$COURT_CROPS_DIR/foot_${f}.png" 2>/dev/null
  # Logo: tight on Red Bull logo (720x280 starting at 600,780)
  ffmpeg -y -i "$COMPOSITED" -vf "select=eq(n\,${f}),crop=720:280:600:780" \
    -vframes 1 "$COURT_CROPS_DIR/logo_${f}.png" 2>/dev/null
done

echo ""
echo "Generated crops:"
ls -1 "$COURT_CROPS_DIR" | sort
echo ""
echo "Done. Review $COURT_CROPS_DIR/"
