---
description: Rules for keeping CLAUDE.md and README.md files up to date after code changes
---

After making any code changes, update the relevant documentation files.

**`CLAUDE.md` files** — context for Claude that is hard to derive from reading the code: non-obvious patterns, gotchas, architectural decisions, naming conventions, and implementation constraints.

**`README.md` files** — human-facing documentation: usage instructions, CLI parameters, key concepts, and examples.

**Rules:**
- Update the file(s) closest to the code that changed.
- If a change spans multiple layers (e.g. a new CLI parameter flows into a core module), update all affected files.
- Do not duplicate content between README and CLAUDE.md — if something is in README, CLAUDE.md should reference it, not repeat it.
- Use prose references (`See README.md for ...`) to point to large human-facing docs, not `@`-imports. `@filename` imports pull the full file into context on every session (real token cost) — reserve them for modular CLAUDE.md sub-files, not READMEs.
- Delete stale entries rather than commenting them out.

**Index of documentation files:**

| File                     | Audience | Contents                                                                   |
|--------------------------|----------|----------------------------------------------------------------------------|
| Root `CLAUDE.md`         | Claude   | Non-obvious architecture behaviors, model building flow, development rules |
| Root `README.md`         | Humans   | Setup, usage, data structure, architecture overview                        |
| `LEGO/CLAUDE.md`         | Claude   | LEGO class API, utilities, standalone functions                            |
| `LEGO/modules/CLAUDE.md` | Claude   | Module interface, execution order, variable/constraint descriptions        |
| `LEGO/helpers/CLAUDE.md` | Claude   | Model comparison and MPS validation utilities                              |
| `research/CLAUDE.md`     | Claude   | Overview and index of research sub-projects                                |
| `research/TR/README.md`  | Humans   | TR usage, key concepts, CLI parameters                                     |
| `research/TR/CLAUDE.md`  | Claude   | Non-obvious TR patterns                                                    |
| `research/MK/README.md`  | Humans   | MK usage, key concepts, CLI parameters                                     |
| `research/MK/CLAUDE.md`  | Claude   | Non-obvious MK patterns                                                    |
| `research/ID/README.md`  | Humans   | ID usage, key concepts, CLI parameters                                     |
| `research/ID/CLAUDE.md`  | Claude   | Non-obvious ID patterns                                                    |