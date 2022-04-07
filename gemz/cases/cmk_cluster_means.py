"""
Demonstrate the effects of centering or not the features in CMK models
"""

from gemz.cases import case

@case
def cmk_cluster_means(output_dir, case, report_path):

    with open(report_path, 'w') as fd:
        fd.write(case)

