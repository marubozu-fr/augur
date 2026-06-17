---
name: fix-issue
description: Fix a GitHub issue end-to-end. Reads the issue, routes to the correct agent via labels, implements the fix, writes tests, and opens a PR.
disable-model-invocation: true
---
Fix the GitHub issue: $ARGUMENTS

## 1. Read the issue

Run `gh issue view $ARGUMENTS --json title,body,labels` to get full details and labels.

## 2. Determine the agent via labels

Read the labels array and look for an `agent-*` label:

- `agent-stats` → use the **stats-dev** agent
- `agent-pipeline` → use the **pipeline-dev** agent
- `agent-frontend` → use the **frontend-dev** agent (Phase 2)

**If no `agent-*` label is found, STOP and ask the user which agent to use.**

## 3. Create a feature branch

```bash
git checkout -b feature/$ARGUMENTS-<short-description>
```

## 4. Execute with the routed agent

1. Read `CLAUDE.md` and `STATUS.md` for current project state
2. Read existing related code to understand current patterns
3. Implement the changes following CLAUDE.md conventions
4. Write tests using the **test-writer** agent
5. Run the full test suite and fix any failures:
   ```bash
   pytest tests/ -v
   ```
6. Run linter and fix any issues:
   ```bash
   ruff check .
   ```

## 5. Update STATUS.md

Update `STATUS.md` to reflect the completed work.

## 6. Commit and open a PR

```bash
git add -A
git commit -m "<type>(<scope>): <description> (closes #$ARGUMENTS)"
git push -u origin feature/$ARGUMENTS-<short-description>
gh pr create --title "<description>" --body "Closes #$ARGUMENTS"
```

Commit type conventions:
- `feat(<scope>)` for new features
- `fix(<scope>)` for bug fixes
- `docs(<scope>)` for documentation
