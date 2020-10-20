""" This contains Thiel analysis plots """
import thiel_analysis
from bokeh.io import curdoc
from bokeh.models.widgets import Div
from bokeh.layouts import column
from scipy.interpolate import interp1d

from config import plot_width, plot_config, colors3
from helper import get_flight_mode_changes
from pid_analysis import Trace, plot_pid_response
from plotting import *
from plotted_tables import get_heading_html

#pylint: disable=cell-var-from-loop, undefined-loop-variable,

def get_thiel_analysis_plots(ulog, px4_ulog, db_data, link_to_main_plots):
    """
    get all bokeh plots shown on the Thiel analysis page
    :return: list of bokeh plots
    """

    page_intro = """
<p>
This page shows the correspondance between a simulated and a real flight log.
</p>
    """
    curdoc().template_variables['title_html'] = get_heading_html(
        ulog, px4_ulog, db_data, None, [('Open Main Plots', link_to_main_plots)],
        'Thiel Analysis') + page_intro

    plots = []
    data = ulog.data_list
    x_range_offset = (ulog.last_timestamp - ulog.start_timestamp) * 0.05
    x_range = Range1d(ulog.start_timestamp - x_range_offset, ulog.last_timestamp + x_range_offset)

    plots = thiel_analysis.startserver


    return plots
