"""
Utils related to report generation
"""

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
