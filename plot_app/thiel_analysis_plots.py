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
    def _resample(time_array, data, desired_time):
        """ resample data at a given time to a vector of desired_time """
        data_f = interp1d(time_array, data, fill_value='extrapolate')
        return data_f(desired_time)

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
    flight_mode_changes = get_flight_mode_changes(ulog)
    x_range_offset = (ulog.last_timestamp - ulog.start_timestamp) * 0.05
    x_range = Range1d(ulog.start_timestamp - x_range_offset, ulog.last_timestamp + x_range_offset)

  #  plots = thiel_analysis.startserver



    for index, axis in enumerate(['roll', 'pitch', 'yaw']):
        axis_name = axis.capitalize()
        data_plot = DataPlot(data, plot_config, 'actuator_controls_0',
                            y_axis_label='[deg/s]', title=axis_name+' Angular Rate',
                            plot_height='small',
                            x_range=x_range)
        data_plot.change_dataset('vehicle_rates_setpoint')
        data_plot.add_graph([lambda data: (axis, np.rad2deg(data[axis]))],
                            colors3[1:2], [axis_name+' Rate Setpoint'],
                            mark_nan=True, use_step_lines=True)
        plot_flight_modes_background(data_plot, flight_mode_changes)

        if data_plot.finalize() is not None: plots.append(data_plot.bokeh_plot)


    data_plot.finalize()
    return plots
