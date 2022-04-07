"""
Entry point to generate reports for Gemz test cases or demos
"""

import os
import argparse
import logging
import webbrowser

import gemz.cases

def status(output_dir):
    cases = gemz.cases.get_cases()
    if not cases:
        print("No demo cases defined")
    else:
        print('Demonstration cases:')
        base_length = max(map(len, cases)) + 4

        for case in cases:
            avail = os.path.exists(
                gemz.cases.get_report_path(output_dir, case)
                )
            print((
                '    '
                + case
                + ''.join([' '] * (base_length - len(case)))
                + ('available' if avail else 'missing')
                ))

def run(output_dir, case):
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
    run(output_dir, case)

    path = gemz.cases.get_report_path(output_dir, case)

    webbrowser.open(f'file://{os.path.abspath(path)}')

def main(args):
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

def add_parser_arguments(parser):
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
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    main(parser.parse_args())
