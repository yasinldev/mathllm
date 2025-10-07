#!/bin/bash
set -e
echo "MathLLM Performance Sprint Setup"
echo "=================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ ! -f .env ]; then
  echo "Creating .env from .env.example..."
  cp .env.example .env
  echo "✓ .env created"
else
  echo "✓ .env already exists"
fi
echo "Making scripts executable..."
chmod +x *.sh *.py 2>/dev/null || true
echo "✓ Scripts are executable"
echo ""
echo "Checking Python dependencies..."
if ! python3 -c "import aiohttp" 2>/dev/null; then
  echo "Installing aiohttp..."
  pip install aiohttp
fi
if ! python3 -c "import requests" 2>/dev/null; then
  echo "Installing requests..."
  pip install requests
fi
echo "✓ Python dependencies ready"
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review/edit .env for your configuration"
echo "2. Start Student server: ./start_student.sh"
echo "3. Wait for startup (30-60s)"
echo "4. Run healthcheck: python3 healthcheck.py http://localhost:8009/v1"
echo "5. Run smoke test: python3 smoke_test.py"
