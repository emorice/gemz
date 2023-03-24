"""
Dashboard to visualize the content of regression files
"""
import json
import yaml

from flask import Flask, render_template, abort
import plotly.graph_objects as go
import plotly.io as pio

from gemz import models
from gemz.cases import get_cases
from gemz.plots.plotly_template import template as plotly_template

app = Flask(__name__)

@app.route('/')
def index():
    """
    List of regression cases
    """
    case_names = list(get_cases())
    return render_template('index.html', cases=case_names)

@app.route('/case/<case_name>')
def _case(case_name):
    """
    Display a specific regression case
    """
    cases = get_cases()

    case = cases.get(case_name)
    if case is None:
        abort(404)

    model_specs = case.model_specs
    return render_template('case.html', case=case_name,
            models=[
                (name, json.dumps(spec, indent=4))
                for name, spec in zip(case.model_unique_names, model_specs)
                ],
            cases=list(cases))

@app.route('/model/<case_name>/<path:model_name>')
def _model(case_name, model_name):
    cases = get_cases()

    path = (
            'tests/test_regressions/' + 
            (f'test_case[{case_name} x {model_name}].yml'
                if model_name != 'all'
                else f'test_case[{case_name}].yml'
                ).translate(
                str.maketrans('[] /', '____')
                )
            )
    with open(path, encoding='utf8') as ifd:
        report = yaml.load(ifd, yaml.CLoader)

    figs = [
            pio.to_html(
                (
                    go.Figure(fig_dict)
                    .update_layout(template=plotly_template)
                    ),
                full_html=False, include_plotlyjs='cdn')
            for fig_dict in report['figures'] 
            ]

    return render_template('model.html', case=case_name, model=model_name,
            cases=list(cases), figures=figs,
            models=cases[case_name].model_unique_names)
