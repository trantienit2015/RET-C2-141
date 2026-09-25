# RET-C2-141 — Retail Staff Motion Capture Performance & Training Q&A Agent

> **Category**: Cat 2 (orchestrates multiple steps to accomplish a specific use case)
> **Industry**: RET

## Overview

Scores a recorded retail staff motion-capture session (for example a customer greeting) and returns
structured training feedback. The input is a JSON object with an action_type (greeting, hand_off
or register_stance) and a session in the open-source freemocap layout: a session_id and a
non-empty list of frames, each with pose_keypoints and a confidence value. The whole payload is
first scanned for staff-identity markers, national ID numbers and credentials; any hit rejects the
request before scoring. The entry node requires a verified external caller.

If more than 20% of the frames have a confidence below 0.6, the agent reports insufficient capture
quality instead of scoring. Otherwise it computes the average bow angle from the nose, shoulder and
hip keypoints of the confident frames and compares it with a 15 degree standard. The result is
turned into a rubric query and matched against a service-standard rubric list supplied through the
graph configuration (kb); no rubric list is bundled, so without one the feedback recommends a manual
trainer review. The feedback record contains the session_id, action type, metric scores with the
gap to the standard, the matched rubric passages, the remediation text and the capture-quality
flag. Before it is returned, the record is checked again for the same staff-identity, ID-number and credential
patterns, and blocked if any are found.

A language model is optional. When a client is supplied through the graph configuration it writes
the remediation text from the metrics and matched rubric; a failed or empty reply ends the run with
an error. Without a client the remediation text is built deterministically from the matched rubric.
The bundled HTTP entry point supplies an Anthropic client only when ANTHROPIC_API_KEY is available.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | 3.11 or later |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent raises
`PlatformRequired` during graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test specification
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
