# Contributing

All changes reach `Main` through a pull request. `Main` should be protected in
Bitbucket (Repository settings > Branch restrictions): no direct pushes, no force
pushes or deletion, at least one approval, and a passing build before merge.

## No direct commits to `Main`

Nobody commits or pushes to `Main` directly: not maintainers, not scripts, not
AI agents. This holds even for small fixes, docs-only changes and work that
has already been reviewed somewhere else.

1. Create a branch from the current `Main` tip (`feature/...`, `fix/...`,
   `docs/...`).
2. Commit to that branch, in a few batched commits.
3. Run `python tools/policy_check.py` and `uv run pytest` and confirm both pass.
4. Open a pull request into `Main` describing what changed, how it was tested,
   and how any conflicts were resolved.
5. A reviewer approves and merges. Authors do not merge their own pull requests
   without that review.

**Agents and API automation.** Tools that write through the Bitbucket API
(for example `createBitbucketRepoCommit` / the `/src` endpoint) must name a
feature branch, never `Main`, and must finish by opening a pull request. An
agent session that finds itself on `Main` stops and creates a branch first.
Agents do not merge pull requests; a person does.

**If a commit lands on `Main` without a pull request**, do not rewrite history
or force-push. Open a follow-up pull request that either reverts it or records
the review it skipped (what changed, test results), and tell the maintainers.
When a pull request is behind `Main`, rebase it onto the current tip and open
the rebased version as a new pull request (or update the existing one) instead
of pushing the merge result to `Main`.

## Repository policy

Everything in this repository must be allowed under Atlassian's
[Acceptable Use Policy](https://www.atlassian.com/legal/acceptable-use-policy) and
[Bitbucket Cloud terms](https://www.atlassian.com/legal/product-specific-terms).
In practice:

- **Source code and small text assets only.** No audio, video, disk images,
  installers, large archives or files over 5 MB. Bitbucket can remove a whole
  repository that uses storage for content unrelated to source code.
- **No secrets.** No keys, tokens, passwords, `.env` files or keystores. Read
  credentials from the environment. If one is committed by mistake, revoke it
  first; deleting the file does not remove it from history.
- **Only code you have the right to share**, under a license compatible with
  MIT. Keep upstream licenses and attributions for anything vendored. No
  proprietary or employer-confidential code, data or model weights.
- **No malicious code**: no exploits, payloads or miners.
- **Honest content.** Results produced by the built-in simulators are
  synthetic; keep them labelled as such (a `_SYNTHETIC` file name or a note in
  the text), never presented as real robot results.
- **Reasonable automation.** Scripts and agents that use the Bitbucket API batch
  their changes into a few commits per PR, stay well inside documented rate
  limits, and never loop on retries.

`tools/policy_check.py` enforces the mechanical parts (secrets, file types and
sizes, malware markers, license file). Bitbucket Pipelines runs it on every pull
request and push to `Main`. Run it locally before you push:

```
python tools/policy_check.py
```

Reviewers check what a script cannot: licensing, confidentiality and whether
content is accurate.
