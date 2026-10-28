"""Install head markup, the explorer placeholder, and public documentation assets."""

import re
from pathlib import Path
from shutil import copyfile

EXPLORER_FALLBACK = re.compile(
    r'<div class="diplodocus-component"'
    r'(?P<attributes>[^>]*data-component-module="[^"]*/optimization-explorer/[^"]+"[^>]*)>'
    r"<div data-component-fallback>"
)
EXPLORER_LOADING = (
    '<div data-explorer-loading role="status">'
    '<span class="explorer-loading-indicator" aria-hidden="true"></span>'
    "<span>Loading the optimization explorer…</span></div>"
)


def main(site: Path | None = None) -> None:
    documentation = Path(__file__).resolve().parents[1] / "docs"
    site = site if site is not None else documentation / "site"
    pages = list(site.rglob("*.html"))
    if not pages:
        raise RuntimeError("Build the documentation before installing public assets.")

    head = (documentation / "head.html").read_text()
    for page in pages:
        original = page.read_text()
        html = EXPLORER_FALLBACK.sub(
            lambda match: (
                '<div class="diplodocus-component explorer-shell"'
                + match["attributes"]
                + "><div data-component-fallback>"
                + EXPLORER_LOADING
            ),
            original,
        )
        if 'class="math math-fallback' in html:
            raise RuntimeError(f"KaTeX could not render an equation in {page}")
        if head not in html:
            if "</head>" not in html:
                raise RuntimeError(f"Missing closing head tag in {page}")
            html = html.replace("</head>", head + "</head>", 1)
        if html != original:
            page.write_text(html)

    for asset in (documentation / "public").iterdir():
        copyfile(asset, site / asset.name)


if __name__ == "__main__":
    main()
