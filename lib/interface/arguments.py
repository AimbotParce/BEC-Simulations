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
    parser.add_argument(
        "-inan",
        "--ignore-nan",
        help="Ignore NaN values in the simulation",
        action="store_true",
        dest="ignoreNan",
    )

    parser.add_argument(
        "-oc",
        "--override-constants",
        help="Override the constants with the provided values (provided as a list of key=value pairs)",
        type=str,
        nargs="+",
        dest="overrideConstants",
    )

    return parser.parse_args()
