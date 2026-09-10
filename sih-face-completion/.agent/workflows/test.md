# /test — Run Tests

1. Run `pytest` — show full output
2. If any tests fail, identify root cause — do not just rerun
3. If coverage <80%, list uncovered functions (`pytest --cov`)
4. Fix only if fix is obviously scoped to the current task
5. If the fix requires changing logic: STOP and ask
6. Report: ✅ X passing, ❌ Y failing, 📊 Z% coverage
