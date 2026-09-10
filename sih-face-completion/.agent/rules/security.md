---
trigger: always_on
description: Security constraints — never bypass
---

# Security Rules
- Never put credentials or API keys in code — use environment variables
- Always validate mesh/image inputs before processing (corrupt file, wrong format, empty mesh)
- Never log full local file paths in any output that could be shared outside the team
- Before installing any package, verify it's actively maintained
- If uncertain about a security decision, STOP and ask
