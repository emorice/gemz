"""
Heterogenous signal-to-noise ratios
"""

from gemz.cases import case
from gemz.reporting import write_header, write_footer

@case
def heterogeneous_snr(_, case_name, report_path):
    """
    Regularized high-dimensional models with variation in SNR between variables
    """

    with open(report_path, 'w', encoding='utf8') as stream:
        write_header(stream, case_name)
        write_footer(stream)
