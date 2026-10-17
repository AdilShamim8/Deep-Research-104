## Summary

<!-- 1-2 sentences. What does this PR do and why? -->

## Linked issues

- Closes #
- Refs #

## Changes

<!-- Bullet list of concrete changes. -->

- Added `X` in `src/Y`
- Changed `Z` to also support `W`
- Removed legacy `Q`

## Public API impact

<!-- How does this affect the public API? Add `BREAKING CHANGE:` if
it breaks an existing endpoint, schema, or env var. -->

- [ ] No public API change
- [ ] New endpoint / schema field
- [ ] Breaking change (called out in commit footer)

## Testing

<!-- How did you verify this? -->

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] New unit tests added
- [ ] Integration test added (if applicable)
- [ ] Manually verified against:
  - [ ] OpenAI provider
  - [ ] DeepSeek provider
  - [ ] Ollama provider
  - [ ] DuckDuckGo search
  - [ ] Streaming endpoint

## Configuration changes

<!-- If you added/changed env vars, list them here. -->

| Env var | Default | Required in prod? | Description |
|---------|---------|-------------------|-------------|
|         |         |                   |             |

## Documentation

- [ ] `README.md` updated
- [ ] `.env.example` updated
- [ ] `CONTRIBUTING.md` updated (if workflow changed)
- [ ] `CHANGELOG.md` entry added

## Operational impact

<!-- Anything the on-call engineer needs to know? -->

- [ ] Adds a new dependency
- [ ] Adds a new long-running background task
- [ ] Changes cache TTLs / invalidation
- [ ] Changes rate-limit defaults
- [ ] Needs a DB migration
- [ ] Requires a config reload (not a restart)

## Checklist

- [ ] Branch is up to date with `main`
- [ ] Commit messages follow Conventional Commits
- [ ] No merge commits (rebase instead)
- [ ] PR title starts with `<type>(<scope>): <subject>`
- [ ] Self-reviewed the diff
- [ ] No secrets / credentials in the diff
