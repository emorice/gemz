"""
Utils related to report generation
"""

import plotly.io as pio

def write_fig(fd, fig):
    """
    Converts a figure to html and write it to the document
    """
    fig_html = pio.to_html(fig,
        full_html=False
        )
    fd.write(fig_html)
