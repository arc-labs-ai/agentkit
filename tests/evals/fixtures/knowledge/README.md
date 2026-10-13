# Knowledge-workflow snapshots

These fixtures support opt-in real-model workflows and deterministic offline acceptance tests. Runs do not need network access to fetch evidence.

## Real sources

- `bank-full.csv` contains **all 45,211 records**, byte-for-byte, from `bank.zip` in the official [UCI Bank Marketing archive](https://archive.ics.uci.edu/dataset/222/bank+marketing). Attribution: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [DOI:10.24432/C5K306](https://doi.org/10.24432/C5K306). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This is historical observational campaign data; the workload does not provide individual financial recommendations.
- `sources.json` contains concise paraphrases of the official [SQLite WAL documentation](https://sqlite.org/wal.html) and [busy-timeout pragma documentation](https://sqlite.org/pragma.html#pragma_busy_timeout), retaining version scope and primary URLs. SQLite documentation is covered by its [public-domain dedication](https://sqlite.org/copyright.html).

## Explicitly synthetic material

The archived team memo in `sources.json` is a stale, contradictory, adversarial test fixture. It is not a published company document. `incident.json` contains ninety synthetic operational events, including baseline, writer contention, checkpoint starvation and recovery. It is not a real customer incident. The research skill is repository-authored procedure guidance. `incident-incomplete.json` removes transaction-owner identifiers and has no approved change; `bank-malformed.csv` intentionally corrupts schema/row shape. Paired scenarios must escalate or return unavailable decisions/metrics instead of filling gaps from model memory.

## Acceptance and maintenance

The analytics oracle independently scans the entire CSV with Python/Decimal; the model uses bounded, read-only SQLite queries. Citation checks require actual query receipts that read the campaign table. A memorized correct answer plus `SELECT 1` fails. Research acceptance requires version-correct decisions, primary evidence identifiers, stale-memo rejection, actual skill/source tool calls, successful evidence handoff, and digest revocation. Support acceptance checks exact log counts and transaction IDs, safe remedies, primary/log citations and an event-triggered peer review.

`PROVENANCE.json` records URLs, licenses, snapshot date, transformation and SHA-256 hashes. To update a snapshot, retrieve it from the listed primary archive, preserve attribution, review changed semantics, regenerate hashes, and rerun the independent fixture/oracle tests. Do not rewrite expected answers merely to match model output. These cases demonstrate specific research, support and analytics behavior; they do not establish reliability for every application or arbitrary data source.
