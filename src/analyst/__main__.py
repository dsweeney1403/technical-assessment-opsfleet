"""Entry point for `python -m analyst`."""

from analyst.cli import app


def main() -> None:
    app(prog_name="python -m analyst")


if __name__ == "__main__":
    main()
