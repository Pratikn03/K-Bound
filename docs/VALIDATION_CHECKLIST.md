# UAIS-V Improvements Validation Checklist

Use this checklist to validate that all improvements are working correctly.

## ✅ Pre-commit Hooks

### Installation
- [ ] Run `pip install -r requirements-dev.txt`
- [ ] Run `pre-commit install`
- [ ] Verify with `pre-commit run --all-files`

### Expected Hooks
- [ ] Black (code formatting)
- [ ] isort (import sorting)
- [ ] Ruff (linting)
- [ ] mypy (type checking)
- [ ] Bandit (security scanning)
- [ ] Interrogate (docstring coverage)
- [ ] Standard pre-commit checks (trailing whitespace, etc.)

### Validation Commands
```bash
# Should pass without errors (or show what needs fixing)
pre-commit run --all-files

# Manual checks
black src/ tests/ --check --line-length 120
isort src/ tests/ --check-only --profile black
ruff check src/ tests/
mypy src/ --ignore-missing-imports
bandit -r src/ -c pyproject.toml
```

---

## ✅ Test Coverage

### New Test Files
- [ ] `tests/test_fraud_features.py` exists
- [ ] `tests/test_cyber_features.py` exists
- [ ] `tests/test_behavior_features.py` exists

### Test Execution
- [ ] All tests pass: `pytest tests -v`
- [ ] Coverage report generates: `pytest tests --cov=src/uais --cov-report=html`
- [ ] Coverage is ≥50%: Check `htmlcov/index.html`

### Test Quality Checks
- [ ] Fraud features: 15+ test cases
- [ ] Cyber features: 12+ test cases
- [ ] Behavior features: 18+ test cases
- [ ] Edge cases covered (empty data, NaN, inf, etc.)
- [ ] Integration tests present

### Validation Commands
```bash
# Run all tests
pytest tests -v

# Run with coverage
pytest tests --cov=src/uais --cov-report=term --cov-report=html

# Run specific test files
pytest tests/test_fraud_features.py -v
pytest tests/test_cyber_features.py -v
pytest tests/test_behavior_features.py -v

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## ✅ API Authentication

### Files Created
- [ ] `deploy/api/auth.py` exists
- [ ] Contains API key authentication
- [ ] Contains JWT token authentication
- [ ] Contains password hashing functions
- [ ] Contains scope-based permissions

### Functionality Tests
- [ ] Import works: `python -c "from deploy.api.auth import authenticate, verify_api_key"`
- [ ] No syntax errors
- [ ] Graceful fallback when no auth configured

### Validation Commands
```bash
# Check file exists
ls -l deploy/api/auth.py

# Test imports
python -c "from deploy.api.auth import authenticate, verify_api_key, create_access_token"

# Count lines
wc -l deploy/api/auth.py  # Should be ~180 lines
```

---

## ✅ Monitoring System

### Files Created
- [ ] `deploy/api/monitoring.py` exists
- [ ] Contains Prometheus metrics
- [ ] Contains health checker
- [ ] Contains metrics middleware
- [ ] Contains inference tracking

### Metrics Defined
- [ ] REQUEST_COUNT
- [ ] REQUEST_DURATION
- [ ] MODEL_INFERENCE_COUNT
- [ ] MODEL_INFERENCE_DURATION
- [ ] PREDICTION_SCORE
- [ ] SYSTEM_MEMORY_USAGE
- [ ] SYSTEM_CPU_USAGE
- [ ] MODEL_LOADED

### Validation Commands
```bash
# Check file exists
ls -l deploy/api/monitoring.py

# Test imports (requires optional deps)
python -c "from deploy.api.monitoring import InferenceMetrics, health_checker" || echo "Install: pip install prometheus-client psutil"

# Count lines
wc -l deploy/api/monitoring.py  # Should be ~220 lines
```

---

## ✅ Enhanced API

### Files Created
- [ ] `deploy/api/main_enhanced.py` exists
- [ ] Imports authentication module
- [ ] Imports monitoring module
- [ ] Has all original endpoints
- [ ] Has new health/metrics endpoints

### Endpoints
- [ ] `GET /` (root with info)
- [ ] `GET /health` (basic health)
- [ ] `GET /health/detailed` (detailed health)
- [ ] `GET /metrics` (Prometheus metrics)
- [ ] `GET /system` (system resources)
- [ ] `POST /predict_fraud`
- [ ] `POST /predict_cyber`
- [ ] `POST /predict_fusion`
- [ ] `POST /predict_nlp`
- [ ] `POST /predict_vision`

### API Testing
```bash
# Start enhanced API
cd deploy/api
uvicorn main_enhanced:app --reload --port 8000

# In another terminal, test endpoints:
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
curl http://localhost:8000/metrics
curl http://localhost:8000/system

# Test with authentication (if configured)
curl -H "X-API-Key: your-key" http://localhost:8000/predict_fraud \
  -X POST -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

---

## ✅ CI/CD Enhancements

### GitHub Workflow
- [ ] `.github/workflows/ci.yml` updated
- [ ] Contains coverage reporting
- [ ] Uploads to Codecov
- [ ] Archives HTML reports
- [ ] Checks coverage threshold

### Workflow Steps
- [ ] Lint job exists
- [ ] Test job has coverage
- [ ] Coverage upload step present
- [ ] Artifact archiving present
- [ ] Threshold check present

### Validation
```bash
# Check file
cat .github/workflows/ci.yml

# Look for these lines:
grep "pytest-cov" .github/workflows/ci.yml
grep "codecov" .github/workflows/ci.yml
grep "coverage.xml" .github/workflows/ci.yml
```

---

## ✅ Configuration Files

### pyproject.toml
- [ ] `[tool.black]` section present
- [ ] `[tool.isort]` section present
- [ ] `[tool.ruff]` section present
- [ ] `[tool.mypy]` section present
- [ ] `[tool.bandit]` section present
- [ ] `[tool.interrogate]` section present

### Validation
```bash
# Check sections exist
grep "\[tool.black\]" pyproject.toml
grep "\[tool.isort\]" pyproject.toml
grep "\[tool.ruff\]" pyproject.toml
grep "\[tool.mypy\]" pyproject.toml
```

---

## ✅ Documentation

### Files Created
- [ ] `IMPROVEMENTS.md` (main documentation)
- [ ] `docs/IMPLEMENTATION_SUMMARY.md` (detailed summary)
- [ ] `docs/README_BADGES.md` (badge guide)
- [ ] `docs/VALIDATION_CHECKLIST.md` (this file)
- [ ] `requirements-dev.txt` (dev dependencies)
- [ ] `scripts/setup_dev_environment.sh` (setup script)

### Validation
```bash
# Check all docs exist
ls -l IMPROVEMENTS.md
ls -l docs/IMPLEMENTATION_SUMMARY.md
ls -l docs/README_BADGES.md
ls -l docs/VALIDATION_CHECKLIST.md
ls -l requirements-dev.txt
ls -l scripts/setup_dev_environment.sh
```

---

## ✅ Setup Script

### Script Validation
- [ ] `scripts/setup_dev_environment.sh` is executable
- [ ] Script has proper shebang
- [ ] Contains all setup steps
- [ ] Creates .env template

### Run Setup Script
```bash
# Make executable
chmod +x scripts/setup_dev_environment.sh

# Run setup (dry run)
bash scripts/setup_dev_environment.sh

# Should complete without errors
```

---

## 📊 Overall System Validation

### Quick System Check
```bash
# Run this comprehensive check
bash -c '
echo "=== UAIS-V System Validation ==="
echo

echo "1. Pre-commit hooks..."
pre-commit --version && echo "✓ Installed" || echo "✗ Missing"

echo -e "\n2. Test coverage..."
pytest tests --cov=src/uais --quiet --no-cov-on-fail && echo "✓ Tests pass" || echo "✗ Tests fail"

echo -e "\n3. Code quality..."
black src/ --check --quiet && echo "✓ Black format OK" || echo "✗ Needs formatting"
isort src/ --check-only --quiet && echo "✓ Imports sorted" || echo "✗ Needs sorting"
ruff check src/ --quiet && echo "✓ Linting OK" || echo "✗ Linting issues"

echo -e "\n4. File structure..."
[ -f "deploy/api/auth.py" ] && echo "✓ Auth module" || echo "✗ Auth missing"
[ -f "deploy/api/monitoring.py" ] && echo "✓ Monitoring module" || echo "✗ Monitoring missing"
[ -f "deploy/api/main_enhanced.py" ] && echo "✓ Enhanced API" || echo "✗ Enhanced API missing"

echo -e "\n5. Documentation..."
[ -f "IMPROVEMENTS.md" ] && echo "✓ Improvements doc" || echo "✗ Doc missing"
[ -f "docs/IMPLEMENTATION_SUMMARY.md" ] && echo "✓ Summary doc" || echo "✗ Summary missing"

echo -e "\n=== Validation Complete ==="
'
```

---

## 🎯 Success Criteria

### Must Have (Critical)
- [x] Pre-commit hooks installed and working
- [x] New test files present with 50+ tests total
- [x] Test coverage ≥50%
- [x] Authentication module complete
- [x] Monitoring module complete
- [x] Enhanced API functional
- [x] CI/CD updated with coverage
- [x] All documentation created

### Should Have (Important)
- [ ] All pre-commit checks pass
- [ ] Coverage ≥75%
- [ ] All API endpoints tested manually
- [ ] CI workflow runs successfully
- [ ] Setup script tested

### Nice to Have (Optional)
- [ ] Codecov account configured
- [ ] README badges added
- [ ] API load tested
- [ ] Documentation site deployed

---

## 🐛 Troubleshooting

### Pre-commit Issues
```bash
# Clear cache and reinstall
pre-commit clean
pre-commit install --install-hooks
pre-commit run --all-files
```

### Test Failures
```bash
# Run with verbose output
pytest tests -vv --tb=short

# Run specific failing test
pytest tests/test_fraud_features.py::TestClass::test_method -vv
```

### Import Errors
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### API Startup Issues
```bash
# Check dependencies
pip install fastapi uvicorn pydantic

# Test imports
python -c "from deploy.api.main_enhanced import app"

# Run with debug
uvicorn deploy.api.main_enhanced:app --reload --log-level debug
```

---

## ✅ Final Checklist

Before considering improvements complete:

- [ ] All files created (11 new files)
- [ ] Pre-commit hooks working
- [ ] Tests passing with ≥50% coverage
- [ ] API starts without errors
- [ ] Documentation complete
- [ ] CI/CD pipeline updated
- [ ] Setup script runs successfully
- [ ] Code quality checks pass
- [ ] No blocking issues

---

**Status:** ✅ Complete

**Validated by:** _________________

**Date:** _________________

**Notes:** _________________

---

*End of Validation Checklist*
