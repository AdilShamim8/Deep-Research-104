---
name: Feature request
about: Propose a new feature for Deep Research 104
title: "[feat] "
labels: ["enhancement", "triage"]
assignees: ["AdilShamim8"]
---

## Summary

<!-- One-paragraph description of the proposed feature. -->

## Motivation

<!-- Why is this needed? Cite a concrete use-case or user pain. -->

## Proposed design

<!-- Sketch the public API and rough internal flow. -->

```python
# If proposing a new API, sketch the call:
from src.deep_research import new_thing
result = new_thing(arg1, arg2=...)
```

## Alternatives considered

<!-- What did you consider and reject? Why? -->

## Out of scope

<!-- Things you explicitly do NOT want bundled with this feature. -->

## Open questions

- [ ] Question 1
- [ ] Question 2

## Size estimate

<!-- Pick one and delete the rest -->
- S — single module change, no API surface change
- M — touches multiple modules, public API change
- L — major subsystem rework

## Definition of done

- [ ] Code merged on `main`
- [ ] Tests added (unit + integration where applicable)
- [ ] `.env.example` updated for any new env vars
- [ ] `README.md` updated for any new public endpoint or CLI command
- [ ] `CONTRIBUTING.md` updated if dev workflow changes
- [ ] CHANGELOG entry
