"""Install shared head markup and public assets in the generated documentation."""

from pathlib import Path
from shutil import copyfile


def main() -> None:
    documentation = Path(__file__).resolve().parents[1] / "docs"
    site = documentation / "site"
    pages = list(site.rglob("*.html"))
    if not pages:
        raise RuntimeError("Build the documentation before installing public assets.")

    head = (documentation / "head.html").read_text()
    for page in pages:
        html = page.read_text()
        if head not in html:
            if "</head>" not in html:
                raise RuntimeError(f"Missing closing head tag in {page}")
            page.write_text(html.replace("</head>", head + "</head>", 1))

    for asset in (documentation / "public").iterdir():
        copyfile(asset, site / asset.name)


if __name__ == "__main__":
    main()
