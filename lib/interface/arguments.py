import argparse


def setupParser():
    parser = argparse.ArgumentParser(
        description="A LaTeX-like markup language for creating documents",
        epilog="This is a work in progress project",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Setup logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file",
        type=str,
        default=None,
    )
    return parser
