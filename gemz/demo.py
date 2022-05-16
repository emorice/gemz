"""
Entry point to generate reports for Gemz test cases or demos
"""

import os
import argparse
import logging
import webbrowser

import gemz.cases

def status(output_dir):
    """
    List available demonstration cases to stdout
    """
    cases = gemz.cases.get_cases()
    if not cases:
        print("No demo cases defined")
    else:
        print('Demonstration cases:')
        base_length = max(map(len, cases)) + 4

        for name, case in cases.items():
            avail = os.path.exists(
                gemz.cases.get_report_path(output_dir, name)
                )
            print((
                ' ' * 4
                + name
                + ''.join([' '] * (base_length - len(name)))
                + ('available' if avail else 'missing')
                ))
            print(*(
                ' ' * 8 + line.strip() + '\n'
                for line in case.__doc__.splitlines()
                if line.strip()
                ), end='', sep='')

def run(output_dir, case):
    """
    Execute a demonstration case to create the case report
    """
    cases = gemz.cases.get_cases()

    if case in cases:
        logging.info('Running case %s', case)

        cases[case](
            output_dir,
            case,
            gemz.cases.get_report_path(output_dir, case)
            )

    else:
        logging.error('No such demonstration case: %s', case)

def show(output_dir, case):
    """
    Run a case, then open the report in the default browser
    """
    run(output_dir, case)

    path = gemz.cases.get_report_path(output_dir, case)

    webbrowser.open(f'file://{os.path.abspath(path)}', new=1)

def main(args):
    """
    Main entry point, execute the action implied by the args.
    """
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if not os.path.exists(args.output_dir):
        logging.info("Creating output directory %s", args.output_dir)
        os.mkdir(args.output_dir)

    if args.status:
        return status(args.output_dir)

    if args.run:
        return run(args.output_dir, args.run)

    if args.show:
        return show(args.output_dir, args.show)

    logging.error('Nothing to do')
    return True

def add_parser_arguments(parser):
    """
    Build the argument parser for the demo entry point
    """
    parser.add_argument('output_dir', help=
        "Directory where to write the demo reports"
        )
    parser.add_argument('--run', help=
        'Demonstration case to run',
        )
    parser.add_argument('--show', help=
        'Demonstration case to run then display in default browser',
        )
    parser.add_argument('--status', help=
        "List existing demo cases",
        action='store_true'
        )
    parser.add_argument('-d', '--debug', help=
        "Enable debug-level logging",
        action='store_true', dest='debug'
        )
    return parser

if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    add_parser_arguments(_parser)
    main(_parser.parse_args())
