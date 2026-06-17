---
name: fix-issue
description: Fix a GitHub issue end-to-end. Reads the issue, routes to the correct agent via labels, implements the fix, writes tests, and opens a PR.
disable-model-invocation: true
---
Fix the GitHub issue: $ARGUMENTS

## 1. Read the issue

Run `gh issue view $ARGUMENTS --json title,body,labels` to get full details and labels.

## 2. Read project state

Read `CLAUDE.md` and `STATUS.md` before any implementation.

## 3. Determine the agent via labels

Read the labels array and look for an `agent-*` label:

- `agent-stats` → use the **stats-dev** agent
- `agent-frontend` → use the **frontend-dev** agent
- `agent-pinescript` → use the **pinescript-dev** agent

**If no `agent-*` label is found, STOP and ask the user which agent to use.**

## 4. Create a feature branch

```bash
git checkout main && git pull
git checkout -b feature/$ARGUMENTS-<short-description>
```

## 5. Execute with the routed agent

1. Read existing related code to understand current patterns
2. Implement the changes following CLAUDE.md conventions
3. Write tests using the **test-writer** agent
4. Run the full test suite and fix any failures:
   ```bash
   pytest tests/ -v
   ```
5. Run linter and fix any issues:
   ```bash
   ruff check .
   ```

## 6. Commit and open a PR

```bash
git add -A
git commit -m "<type>(<scope>): <description> (closes #$ARGUMENTS)"
git push -u origin feature/$ARGUMENTS-<short-description>
gh pr create --title "<description>" --body "Closes #$ARGUMENTS"
```
