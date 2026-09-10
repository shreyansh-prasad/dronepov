# /review — Code Review

1. Run `git diff main` to identify changed files
2. For each changed file check:
   - [ ] No hardcoded secrets or local file paths
   - [ ] Error handling on all new functions (no bare `except:`)
   - [ ] No `print()` in pipeline code
   - [ ] Type hints on all new function signatures
   - [ ] Every output mesh carries its provenance tags before export
3. Run `ruff check .` — report errors
4. Check: were any files changed that are outside the stated task scope?
5. Output summary table: File | Change Type | Issues Found
6. If critical issues found: list them, ask if I want them fixed
