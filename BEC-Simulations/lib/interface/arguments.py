import argparse


def setupParser(parseArgs=True):
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
        "-o",
        "--output",
        help="Output folder for the simulation",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-a",
        "--animate",
        help="Show the animation of the simulation",
        action="store_true",
        dest="animate",
    )

    parser.add_argument(
        "-cn",
        "--crank-nicolson",
        help="Python file with the Crank-Nicolson module",
        type=str,
        required=False,
        default=None,
        dest="CNmodule",
    )

    parser.add_argument(
        "-sp",
        "--show-parts",
        help="Show imaginary and real parts of the wave function",
        action="store_true",
        dest="showParts",
    )

    parser.add_argument(
        "-t",
        "--show-theoretical",
        help="Show the theoreical time dependent wave function in the animation",
        action="store_true",
        dest="theoretical",
    )

    # parser.add_argument(
    #     "-inan",
    #     "--ignore-nan",
    #     help="Ignore NaN values in the simulation",
    #     action="store_true",
    #     dest="ignoreNan",
    # )

    parser.add_argument(
        "-oc",
        "--override-constants",
        help="Override the constants with the provided values (provided as a list of key=value pairs)",
        type=str,
        nargs="+",
        dest="overrideConstants",
    )

    parser.add_argument(
        "-cpu",
        "--cpu-only",
        help="Run the simulation only on the CPU",
        action="store_true",
        dest="cpuOnly",
    )

    parser.add_argument(
        "-s",
        "--simple-text",
        dest="simpleText",
        help="Do not add colors to the text",
        action="store_true",
    )

    if parseArgs:
        args = parser.parse_args()

        if args.overrideConstants:
            res = {}
            for constant in args.overrideConstants:
                if "=" not in constant:
                    raise ValueError(f"Invalid constant override {constant}. Missing '='")
                key, value = constant.split("=")
                res[key] = value  # These are strings!!!
            args.overrideConstants = res
        return args
    else:
        return parser
