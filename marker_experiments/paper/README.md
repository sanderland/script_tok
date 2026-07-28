# ACL-format source

`acl_latex.tex` + `custom.bib`. Content mirrors `../paper.md`, which stays the working draft.

## Building

The ACL style files are **not** included and could not be fetched from this environment
(GitHub raw returned 404, the API was unreachable). Download them into this directory:

```
https://github.com/acl-org/acl-style-files
  -> acl.sty
  -> acl_natbib.bst
```

Then:

```
pdflatex acl_latex
bibtex acl_latex
pdflatex acl_latex
pdflatex acl_latex
```

**This has never been compiled.** There is no TeX toolchain in the container, so the source
is unverified: expect to fix at least the usual first-build complaints (missing style file,
unicode in the Cyrillic/Greek examples, table widths in the two-column layout). The Cyrillic
and Greek examples in §3.1 are written with ASCII placeholders (`<Cyr>`, `\Delta`, `\pi`)
rather than literal glyphs precisely because `pdflatex` will not typeset them without extra
packages; switch to `xelatex` with a Unicode font if you want the real characters.

Remove `[review]` from `\usepackage[review]{acl}` for a camera-ready build.

## Bibliography status

**Verify every entry before submission.** This environment had no access to the ACL
Anthology or any bibliographic database, so nothing in `custom.bib` was resolved against an
authoritative source. Entries carry one of three markers:

- `[repo]` — cited in this repository's own source, so venue and identifiers come from there
  (PathPiece/MI-pruning, Goldfish)
- `[standard]` — widely cited work whose details are believed correct but were not checked
- `[check]` — reconstructed from memory; author lists, venues, years and page numbers are
  the most likely to be wrong

The `[check]` entries are GPT-2, Tokenization and the Noiseless Channel, FineWeb, FineWiki,
CodeParrot, Rosetta Code, and Llama 2. A bibliography with plausible-looking wrong entries is
worse than an incomplete one, so these are flagged rather than presented as settled.

## Content differences from `paper.md`

- Adds §1 Introduction, which the Markdown draft lacks, framing the contribution as the
  choice of *which* units to delimit rather than the marker mechanism itself.
- Adds citations throughout; the Markdown draft has none.
- Tables are `booktabs`; the six-language and MinGram tables are unchanged numerically.
- The mixed prose+code results are compressed to a paragraph (§5.4) since they come from the
  earlier per-script variant.
- Author block is `Anonymous`; acknowledgments are a placeholder.
