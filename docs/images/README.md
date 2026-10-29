# Public interface screenshots

Captured on 2026-10-29 from the local Reddit-readiness candidate on
`chore/reddit-launch-readiness-v1`, based on public commit
`4c3987e669360e36930ffe8d4317dd040bc0f47c`. These captures include the candidate's
uncommitted first-run and navigation improvements; they are not screenshots of
unchanged public main. The source-only app used the accepted locked Python 3.14.10,
Streamlit 1.66.1 and pandas 3.0.6 environment on loopback, with no model artifacts.

- `overview.jpg`: navigation, method count, unavailable-model count and research status.
- `catalog.jpg`: public method descriptions and missing local weights.
- `demo.jpg`: original synthetic geometry, validation rows and null prediction fields.
- `human-review.jpg`: an actual saved synthetic observation, one reviewed example,
  one pending example and explicit review exports. The note is invented solely for
  this demonstration; no model prediction or verified label exists.
- `exports.jpg`: actual synthetic validation rows and CSV/JSON download actions.

These are actual browser captures, not generated product mockups. All five contain
only the public interface or original geometry; no dataset imagery, real inference,
workstation paths, account details or browser chrome. Each is 1280 x 720 pixels and
under 130 KB; all five total approximately 546 KiB. The original pixels were
re-encoded as optimized JPEGs without EXIF, GPS, XMP, ICC profiles, comments or
embedded thumbnails. The review time and opaque demo identifiers are generated
synthetic-session evidence, not personal identifiers. FactoryVision-owned UI and synthetic content use the
repository MIT licence. Dependency terms remain separate (see ../licensing.md).

The images are repository presentation assets and are excluded from wheel and sdist
packages. README image URLs are absolute so packaged documentation can refer to the
public images without bundling binary assets.
The candidate images must reach public main before the updated absolute URLs
reflect this gallery; local review uses the candidate files. Raw captures and
browser diagnostics remain ignored and are not publication assets.
