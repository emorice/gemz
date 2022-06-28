"""
Utils related to report generation
"""

from contextlib import contextmanager

import plotly.io as pio

def write_fig(stream, *figs):
    """
    Converts a figure to html and write it to the document
    """
    for fig in figs:
        fig_html = pio.to_html(fig,
            full_html=False
            )
        stream.write(fig_html)

def write_header(stream, case_name):
    """
    Writes a basic html header

    Should be replaced with templates at some point.
    """
    print(
        '<!doctype html><html>'
        '<head><title>'
        + case_name
        + '</title></head>'
        '<body><h1>'
        + case_name
        + '</h1>',
        file=stream
        )

def write_footer(stream):
    """
    Writes a basic html footer
    """
    print('</body></html>', file=stream)

@contextmanager
def open_report(report_path, case_name):
    """
    Open an html file and adds header and footer
    """
    with open(report_path, 'w', encoding='utf8') as stream:
        write_header(stream, case_name)
        yield stream
        write_footer(stream)
