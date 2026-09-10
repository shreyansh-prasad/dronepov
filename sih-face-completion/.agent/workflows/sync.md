# /sync — Update Project Memory

Run this after EVERY completed task. This is how decisions persist across sessions.

## Steps
1. Read @CONTEXT.md
2. Identify what changed in this task:
   - New files created
   - Packages installed
   - Architectural decisions made
   - Functionality added or modified
   - Bugs fixed
3. Update CONTEXT.md:
   - Add new packages to "Installed Packages"
   - Add decisions to "Active Decisions" with reason + date
   - Update "Current Functionality" checklist
   - Update "In Progress" section
   - Remove completed items from "In Progress"
   - Add newly discovered issues to "Known Issues"
   - Set "Last Updated" to today's date + task name
4. Write the updated CONTEXT.md back to disk
5. Output: "✅ CONTEXT.md synced — [X] changes recorded"

## What to NEVER remove from CONTEXT.md
- Active Decisions (these are permanent — agent must not reverse them)
- Current Functionality checklist items (only add, never remove unless a feature was deleted)
