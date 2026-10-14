"""Entry point for `python -m analyst`. The CLI arrives in phase 1."""

from analyst import __version__


def main() -> None:
    print(f"OpsFleet Analyst {__version__}")


if __name__ == "__main__":
    main()
