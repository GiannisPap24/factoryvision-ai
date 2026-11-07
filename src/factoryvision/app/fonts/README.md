# IBM Plex Sans — local interface fonts

Original TTF files from the official
[@ibm/plex-sans@1.1.0 release](https://github.com/IBM/plex/releases/tag/%40ibm%2Fplex-sans%401.1.0),
`ibm-plex-sans.zip`, `fonts/complete/ttf/`. Internal font version: **3.005**.
The selected files are Regular (400), Medium (500), SemiBold (600), Bold (700),
Italic (400) and SemiBoldItalic (600). No outlines, names or glyphs were modified.

These files remain under [SIL Open Font License 1.1](LICENSE.txt), with the original
IBM copyright and Reserved Font Name **Plex**. Redistribution must retain that
notice and license; the fonts may not be sold by themselves. Modified derivatives
must respect reserved-name restrictions and remain under the OFL. FactoryVision's
MIT license does not relicense these font files. The notice is preserved verbatim.

[manifest.json](manifest.json) pins every font and the license by SHA-256.
Distribution inspection permits only the exact paths and hashes, with the original
license. Other binaries and altered font bytes remain rejected.

English letters, modern Greek including tonos/dialytika, digits, decimal/scientific
notation, micro/mu, degree, ohm, plus/minus, multiplication/division, comparisons,
minus, arrows, powers and infinity were checked against the Unicode cmap. Some
polytonic Greek forms are absent; the native Source Sans/sans-serif stack handles
missing glyphs. This is not a claim of universal Unicode or mathematical coverage.

The standard-library loader embeds these fixed public bytes as CSS data URLs.
There is no remote font request, OS font installation, `local()` lookup, JavaScript
font loader or enabled static-file server. The native Streamlit theme selects the
family, with local fallback text while the browser loads it (`font-display: swap`).
