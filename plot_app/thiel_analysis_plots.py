""" This contains Thiel analysis plots """
import thiel_analysis
from bokeh.io import curdoc
from bokeh.models.widgets import Div
from bokeh.layouts import column
from scipy.interpolate import interp1d

from plotting import *
from plotted_tables import *
from configured_plots import generate_plots

from config import *
from helper import *
from leaflet import ulog_to_polyline
from plotted_tables import get_heading_html

#pylint: disable=cell-var-from-loop, undefined-loop-variable,

def get_thiel_analysis_plots(ulog, px4_ulog, db_data, vehicle_data, link_to_main_plots):
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
        ulog, px4_ulog,db_data, None, [('Open Main Plots', link_to_main_plots)],
        'Thiel Analysis') + page_intro

    plots = []
    data = ulog.data_list
    flight_mode_changes = get_flight_mode_changes(ulog)
    x_range_offset = (ulog.last_timestamp - ulog.start_timestamp) * 0.05
    x_range = Range1d(ulog.start_timestamp - x_range_offset, ulog.last_timestamp + x_range_offset)

  #  plots = thiel_analysis.startserver

# plot positions

# Position plot
    data_plot = DataPlot2D(data, plot_config, 'vehicle_local_position',
                           x_axis_label='[m]', y_axis_label='[m]', plot_height='large')
    data_plot.add_graph('y', 'x', colors2[0], 'Estimated',
                        check_if_all_zero=True)
    if not data_plot.had_error: # vehicle_local_position is required
        data_plot.change_dataset('vehicle_local_position_setpoint')
        data_plot.add_graph('y', 'x', colors2[1], 'Setpoint')
        # groundtruth (SITL only)
        data_plot.change_dataset('vehicle_local_position_groundtruth')
        data_plot.add_graph('y', 'x', color_gray, 'Groundtruth')
        # GPS + position setpoints
        plot_map(ulog, plot_config, map_type='plain', setpoints=True,
                 bokeh_plot=data_plot.bokeh_plot)
        if data_plot.finalize() is not None:
            plots.append(data_plot.bokeh_plot)

            # Leaflet Map
            try:
                pos_datas, flight_modes = ulog_to_polyline(ulog, flight_mode_changes)
                curdoc().template_variables['pos_datas'] = pos_datas
                curdoc().template_variables['pos_flight_modes'] = flight_modes
            except:
                pass
            curdoc().template_variables['has_position_data'] = True


    
    # Local position
    for axis in ['x', 'y', 'z']:
        data_plot = DataPlot(data, plot_config, 'vehicle_local_position',
                             y_axis_label='[m]', title='Local Position '+axis.upper(),
                             plot_height='small', x_range=x_range)
        data_plot.add_graph([axis], colors2[0:1], [axis.upper()+' Estimated'], mark_nan=True)
        data_plot.change_dataset('vehicle_local_position_setpoint')
        data_plot.add_graph([axis], colors2[1:2], [axis.upper()+' Setpoint'],
                            use_step_lines=True)
        plot_flight_modes_background(data_plot, flight_mode_changes)

        if data_plot.finalize() is not None: plots.append(data_plot)
    
    x_range_offset = (ulog.last_timestamp - ulog.start_timestamp) * 0.05
    x_range = Range1d(ulog.start_timestamp - x_range_offset, ulog.last_timestamp + x_range_offset)


    # exchange all DataPlot's with the bokeh_plot and handle parameter changes

    jinja_plot_data = []
    for i in range(len(plots)):
        if plots[i] is None:
            plots[i] = column(param_changes_button, width=int(plot_width * 0.99))
        if isinstance(plots[i], DataPlot):
            if plots[i].param_change_label is not None:
                param_change_labels.append(plots[i].param_change_label)

            plot_title = plots[i].title
            plots[i] = plots[i].bokeh_plot

            fragment = 'Nav-'+plot_title.replace(' ', '-') \
                .replace('&', '_').replace('(', '').replace(')', '')
            jinja_plot_data.append({
                'model_id': plots[i].ref['id'],
                'fragment': fragment,
                'title': plot_title
                })

    curdoc().template_variables['plots'] = jinja_plot_data
    return plots
