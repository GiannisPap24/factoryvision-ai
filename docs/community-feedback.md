# Community feedback

Use [GitHub Issues](https://github.com/GiannisPap24/factoryvision-ai/issues/new/choose)
for ordinary non-sensitive bug reports, feature requests and documentation feedback.
For vulnerabilities, use the private route in [SECURITY.md](../SECURITY.md).
Public issues are public records; confidentiality or removal of all copies is not promised.

## Useful reports

- **Installation:** source commit, OS/architecture, Python version, the documented
  step attempted and a minimal sanitized error description.
- **Application behavior:** workspace, intended task, expected and observed behavior,
  and a reproduction using the built-in synthetic demo or described synthetic inputs.
- **Features or documentation:** the task that is difficult, the relevant section
  or workspace and the proposed improvement. Distinguish existing behavior from a request.

Do not attach credentials, customer information, private dataset images, model
weights, sensitive logs or exploit details. Do not include full environment dumps
or personal contact details. Missing weights and disabled real inference are expected
in a fresh checkout; follow [setup and troubleshooting](local-demo.md).

## Triage

The maintainer checks whether a report is a duplicate, needs information, reproduces
on the supported platform, or concerns documentation. Proposed fixes are verified
before closure and linked to the report. These are working states, not automated
labels or a response-time commitment. A scientific proposal needs its own protocol;
feedback is not a ground-truth annotation and never triggers automatic training.

If sensitive information appears in a public report, do not quote or redistribute
it. Move the discussion to the appropriate private route and use GitHub's applicable
moderation/removal process without promising that public copies can be recalled.

## First-month tracking

The observation period starts on the actual first community announcement date,
not the source publication date. Record only voluntary, relevant evidence from
public feedback and authorized discussions. No telemetry, tracking pixels or
new external analytics service is introduced.

| Measure | Counting rule |
|---|---|
| Confirmed external testers | Distinct non-maintainers explicitly reporting that they tried a documented workflow; record outcome and public evidence, and deduplicate repeat reports. Stars, forks and page views are not users. |
| Useful feedback | Distinct actionable observations, such as a reproducible failure or a specific confusing step; a report may contain several observations. |
| Installation issues | Distinct setup problems reported; mark confirmed, unconfirmed or resolved and distinguish unsupported environments. |
| Bug reports | Distinct reported behavior defects; keep reported and reproduced counts separate. |
| Feature requests | Distinct requested capabilities, with duplicates linked; requests are not implementation commitments. |
| Industrial use-case discussions | Voluntary or explicitly authorized substantive discussions about a task and imaging constraints; interest is not a customer, pilot or validated use case. |

Review once per week during that first month. Keep a minimal local ledger with an
opaque record ID, date, category, public evidence URL or non-identifying authorized
summary, reproduction status, next action and resolution. Use a separate opaque
tester key only when needed to deduplicate; do not collect emails, employers,
private images or customer data. Keep identity links only while needed for the
active feedback round; review and remove unnecessary local identity mappings at
the first-month review. Prefer aggregate summaries afterwards. Public GitHub
records have their own retention and cannot be erased by clearing a local ledger.

Before any announcement, these measures are **not yet observed**, not measured
zero usage. Record missing observations explicitly. Do not infer adoption or
industrial reliability from a small voluntary sample. In-app human review and
annotation contracts belong to the separately scoped 5A milestone.
