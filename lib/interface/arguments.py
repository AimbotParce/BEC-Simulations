import argparse


def setupParser():
    parser = argparse.ArgumentParser(
        description="Run the Crank-Nicolson simulation for the Gross-Pitaevskii equation",
        epilog="This is for a physics' final degree project at the University of Barcelona",
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
        help="Python file with the wave function and potential functions",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-t",
        "--show-theoretical",
        help="Show the theoretical wave function",
        action="store_true",
        dest="showTheoretical",
    )
    return parser.parse_args()
