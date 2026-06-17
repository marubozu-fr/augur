---
name: pr-ready
description: Run all quality checks before opening a PR. Linting, tests, and code review.
disable-model-invocation: true
---
Run the full quality pipeline on the current branch before creating a PR.

1. **Lint check**
   - `ruff check .`
   - Fix any issues found

2. **Type check** (if TypeScript files exist)
   - `cd frontend && pnpm tsc --noEmit`

3. **Test suite**
   - `pytest tests/ -v`
   - All tests must pass. Fix failures before proceeding.

4. **Code review**
   - Use the **code-reviewer** agent to review the diff: `git diff main...HEAD`
   - Address all MUST FIX findings

5. **Statistical review** (if stat modules changed)
   - Verify every stat has a baseline method
   - Verify pending samples are excluded
   - Verify sample sizes are reported

6. **STATUS.md updated**
   - Confirm `STATUS.md` reflects the work done in this branch

7. **Summary**
   - List all files changed
   - List all tests added/modified
   - List any remaining findings intentionally deferred
   - Confirm the branch is ready for PR
