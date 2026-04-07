#!/bin/bash
# Setup development environment for VRT project

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up VRT development environment..."

# Install git hooks
if [ -d ".github/hooks" ]; then
    echo "Installing git hooks..."
    cp .github/hooks/* .git/hooks/ 2>/dev/null || true
    chmod +x .git/hooks/* 2>/dev/null || true
    echo "  ✓ Git hooks installed"
fi

# Check Python environment
if command -v python &>/dev/null; then
    echo "  ✓ Python: $(python --version 2>&1)"
else
    echo "  ⚠ Python not found"
fi

# Check for required packages
echo "Checking dependencies..."
python -c "import vapoursynth" 2>/dev/null && echo "  ✓ vapoursynth" || echo "  ⚠ vapoursynth not installed"
python -c "import torch" 2>/dev/null && echo "  ✓ torch" || echo "  ⚠ torch not installed"
python -c "import av" 2>/dev/null && echo "  ✓ av (PyAV)" || echo "  ⚠ av not installed"

echo ""
echo "Setup complete. See README.md for usage."
