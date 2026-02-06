#!/bin/bash
# Development environment setup script for UAIS-V
# This script installs development tools and configures the environment

set -e  # Exit on error

echo "🚀 UAIS-V Development Environment Setup"
echo "========================================"
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

echo "📦 Step 1: Installing development dependencies..."
pip install -r requirements-dev.txt
echo -e "${GREEN}✓${NC} Development dependencies installed"
echo

echo "🔧 Step 2: Setting up pre-commit hooks..."
pre-commit install
echo -e "${GREEN}✓${NC} Pre-commit hooks installed"
echo

echo "🧪 Step 3: Running initial code quality checks..."
echo

echo "  → Running Black formatter..."
black src/ tests/ dashboard/ --line-length 120 --check || {
    echo -e "${YELLOW}⚠${NC} Code formatting issues found. Run: black src/ tests/ --line-length 120"
}
echo

echo "  → Running isort..."
isort src/ tests/ dashboard/ --check-only --profile black || {
    echo -e "${YELLOW}⚠${NC} Import sorting issues found. Run: isort src/ tests/ --profile black"
}
echo

echo "  → Running Ruff linter..."
ruff check src/ tests/ || {
    echo -e "${YELLOW}⚠${NC} Linting issues found. Run: ruff check src/ tests/ --fix"
}
echo

echo "  → Running mypy type checker..."
mypy src/ --ignore-missing-imports || {
    echo -e "${YELLOW}⚠${NC} Type checking issues found (non-critical)"
}
echo

echo "📊 Step 4: Running tests with coverage..."
pytest tests --cov=src/uais --cov-report=term --cov-report=html -v || {
    echo -e "${YELLOW}⚠${NC} Some tests failed"
}
echo

echo "📋 Step 5: Generating coverage report..."
if [ -d "htmlcov" ]; then
    echo -e "${GREEN}✓${NC} Coverage report generated at: htmlcov/index.html"
fi
echo

echo "🔐 Step 6: Environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env <<EOF
# UAIS-V Environment Configuration
# Copy this file to .env and customize as needed

# API Authentication (optional - leave empty for development mode)
UAIS_SECRET_KEY=
UAIS_API_KEYS=

# MLflow Configuration
MLFLOW_TRACKING_URI=./src/mlruns

# Logging
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✓${NC} Created .env template file"
else
    echo -e "${YELLOW}⚠${NC} .env file already exists (skipping)"
fi
echo

echo "📚 Step 7: Setup summary"
echo "========================"
echo
echo "Development tools installed:"
echo "  • Black (code formatter)"
echo "  • isort (import sorter)"
echo "  • Ruff (linter)"
echo "  • mypy (type checker)"
echo "  • Bandit (security scanner)"
echo "  • pytest + pytest-cov (testing)"
echo "  • pre-commit (git hooks)"
echo
echo "Pre-commit hooks configured:"
echo "  • Will run automatically on git commit"
echo "  • Format code, sort imports, lint, type check"
echo "  • Run manually: pre-commit run --all-files"
echo
echo "Next steps:"
echo "  1. Review .env file and set API keys if needed"
echo "  2. Run tests: pytest tests -v"
echo "  3. Start development server: uvicorn deploy.api.main_enhanced:app --reload"
echo "  4. View coverage report: open htmlcov/index.html"
echo
echo -e "${GREEN}✅ Development environment setup complete!${NC}"
echo
echo "Useful commands:"
echo "  • Format code:      black src/ tests/ --line-length 120"
echo "  • Sort imports:     isort src/ tests/ --profile black"
echo "  • Lint:             ruff check src/ tests/"
echo "  • Type check:       mypy src/ --ignore-missing-imports"
echo "  • Run tests:        pytest tests -v"
echo "  • Test coverage:    pytest tests --cov=src/uais --cov-report=html"
echo "  • Pre-commit:       pre-commit run --all-files"
echo
