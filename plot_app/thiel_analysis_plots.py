""" This contains Thiel analysis plots """


# TO DO: change all the data to the form dfsim.metric


import px4tools
import numpy as np
import math
import io
import os
import sys
import errno

#import thiel_analysis
from bokeh.io import curdoc,output_file, show
from bokeh.models.widgets import Div
from bokeh.layouts import column
from scipy.interpolate import interp1d

import plotting
from plotted_tables import *
from configured_plots import *
from os.path import dirname, join

from config import *
from helper import *
from leaflet import ulog_to_polyline
from bokeh.models import CheckboxGroup
from bokeh.models import RadioButtonGroup
from bokeh.models.widgets import FileInput
from bokeh.models.widgets import Paragraph

import pandas as pd
import argparse

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.application.handlers import DirectoryHandler


#pylint: disable=cell-var-from-loop, undefined-loop-variable,

DATA_DIR = join(dirname(__file__), 'datalogs')

DEFAULT_FIELDS = ['XY', 'LatLon', 'VxVy']

STANDARD_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

simname = 'airtonomysim.ulg'
realname = 'airtonomyreal.ulg'
sim_polarity = 1  # determines if we should reverse the Y data
real_polarity = 1
simx_offset = 0
realx_offset = 0
read_file = True
reverse_sim_data = False
reverse_real_data = False
new_data = True
metric = 'x'

sim_reverse_button = RadioButtonGroup(
        labels=["Sim Default", "Reversed"], active=0)
sim_reverse_button.on_change('active', lambda attr, old, new: reverse_sim())
real_reverse_button = RadioButtonGroup(
        labels=["Real Default", "Reversed"], active=0)
real_reverse_button.on_change('active', lambda attr, old, new: reverse_real())

# set up widgets

stats = PreText(text='Thiel Coefficient', width=500)
# datatype = Select(value='XY', options=DEFAULT_FIELDS)




@lru_cache()
def load_data(filename):
    fname = join(DATA_DIR, filename)
    ulog = load_ulog_file(fname)
    cur_dataset = ulog.get_dataset('vehicle_local_position')
    return cur_dataset


@lru_cache()
def get_data(simname,realname, metric):
    dfsim = load_data(simname)
    sim_data = dfsim.data[metric]
    pd_sim = pd.DataFrame(sim_data, columns = ['sim'])
    sim_time = dfsim.data['timestamp']
    pd_time = pd.DataFrame(sim_time, columns = ['time'])
    dfreal = load_data(realname)
    real_data = dfreal.data[metric]
    pd_real = pd.DataFrame(real_data, columns = ['real'])
    new_data = pd.concat([pd_time,pd_sim, pd_real], axis=1)
    new_data = new_data.dropna()   # remove missing values
    return new_data
    # print(new_data)
    # dfdata = pd.DataFrame(cur_dataset.data) 
    # data = pd.concat([dfsim, dfreal], axis=1)
    # data = data.dropna()   # remove missing values
    # sim_mean = dfsim.y.mean()  # get the average
    # real_mean = dfreal.y.mean()
    # mean_diff = sim_mean - real_mean 
    # data.realy = data.realy + mean_diff # normalize the two
    # data['sim'] = dfsim.x
    # data['simt'] = dfsim.timestamp
    # data['real'] = dfreal.x
    # data['realt'] = dfreal.timestamp


def update(selected=None):
    global read_file, reverse_sim_data, reverse_real_data, new_data, datalog, original_data, new_data, datasource
    if (read_file):
        original_data = get_data(simname, realname, metric)
        datalog = copy.deepcopy(original_data)
        read_file = False
    print("Sim offset", simx_offset)
    print("Real offset", realx_offset)
    if reverse_sim_data:
        datalog[['sim']] = sim_polarity * original_data['sim']  # reverse data if necessary
        simmax = round(max(datalog[['sim']].values)[0])  # reset the axis scales as appopriate (auto scaling doesn't work)
        simmin = round(min(datalog[['sim']].values)[0])
        datasource.data = datalog
        reverse_sim_data = False
    if reverse_real_data:
        datalog['real'] = real_polarity * original_data['real']
        realmax = round(max(datalog[['real']].values)[0])
        realmin = round(min(datalog[['real']].values)[0])
        datasource.data = datalog
        reverse_real_data = False
    if new_data:
        datasource.data = datalog
        new_data = False


def upload_new_data_sim(attr, old, new):
    global simname
    decoded = b64decode(new)
    simname = io.BytesIO(decoded)
    update()

def upload_new_data_real(attr, old, new):
    global realname
    decoded = b64decode(new)
    realname = io.BytesIO(decoded)
    update()

def update_stats(data):
    real = np.array(data['realy'])
    sim = np.array(data['simy'])
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for n in range(len(real)):
        sum1 = sum1 + (real[int(n)]-sim[int(n)])**2
        sum2 = sum2 + real[int(n)]**2
        sum3 = sum3 + sim[int(n)]**2
    sum1 = 1/len(real) * sum1
    sum2 = 1/len(real) * sum2
    sum3 = 1/len(real) * sum3
    sum1 = math.sqrt(sum1)
    sum2 = math.sqrt(sum2)
    sum3 = math.sqrt(sum3)
    TIC = sum1/(sum2 + sum3)
    stats.text = 'Thiel coefficient: ' + str(round(TIC,3))


def reverse_sim():
    global sim_polarity, reverse_sim_data
    if (sim_reverse_button.active == 1): sim_polarity = -1
    else: sim_polarity = 1
    reverse_sim_data = True
    update()

def reverse_real():
    global real_polarity, reverse_real_data
    if (real_reverse_button.active == 1): real_polarity = -1
    else: real_polarity = 1
    reverse_real_data = True
    update()

def change_sim_scale(shift):
    global simx_offset, new_data
    simx_offset = shift
    new_data = True
    update()

def change_real_scale(shift):
    global realx_offset, new_data
    realx_offset = shift
    new_data = True
    update()



def sim_change(attrname, old, new):
    global metric
    print("Sim change:", new)
    print(dfdata[new])
    metric = new
    update()   

def get_thiel_analysis_plots(ulog, px4_ulog, db_data, vehicle_data, link_to_main_plots):
    global datalog, original_data,datasource
    """
    get all bokeh plots shown on the Thiel analysis page
    :return: list of bokeh plots
    """
    def _resample(time_array, data, desired_time):
        """ resample data at a given time to a vector of desired_time """
        data_f = interp1d(time_array, data, fill_value='extrapolate')
        return data_f(desired_time)


    sim = False
    if link_to_main_plots.find("sim") is not -1:
        temp_link_to_main_plots = link_to_main_plots.replace('sim','')
        sim = True


    if sim:
        print("do some sim stuff")
    else:
        print("do regular stuff")
        page_intro = """
    <p>
    This page shows the correspondance between a simulated and a real flight log.
    </p>
        """
        # curdoc().template_variables['title_html'] = get_heading_html(
        #     ulog, px4_ulog,db_data, None, [('Open Main Plots', link_to_main_plots,)],
        #     'Thiel Analysis') + page_intro
        curdoc().template_variables['title_html'] = get_heading_html(
            ulog, px4_ulog, db_data, None,
            additional_links=[('Open Main Plots', link_to_main_plots,),("Open Matching Simulation Log", '/browse?search=sim')])



      
        keys = []
        data = ulog.data_list
        for d in data:
            data_keys = [f.field_name for f in d.field_data]
            data_keys.remove('timestamp')
            keys.append(data_keys)



        datalog = get_data(simname, realname, metric)
        original_data = copy.deepcopy(datalog)



#        print(datalog)

        # data2 = data['x']

        # t = realsource['timestamp']
        # x = realsource['x']
        # y = realsource['y']

                # set up plots

        # simsource = ColumnDataSource(data = dict(simt=[],sim=[]))
        # realsource = ColumnDataSource(data = dict(realt=[],real=[]))


        datatype = Select(value='x', options=keys[0])

        datatype.on_change('value', sim_change)

    

        file_input = FileInput(accept=".ulg")
        file_input.on_change('value', upload_new_data_sim)
        file_input2 = FileInput(accept=".ulg")
        file_input2.on_change('value', upload_new_data_real)

        intro_text = Div(text="""<H2>Sim/Real Thiel Coefficient Calculator</H2>""",width=500, height=100, align="center")
        sim_upload_text = Paragraph(text="Upload a simulator datalog:",width=500, height=15)
        real_upload_text = Paragraph(text="Upload a corresponding real-world datalog:",width=500, height=15)
        choose_field_text = Paragraph(text="Choose a data field to compare:",width=500, height=15)
        #checkbox_group = CheckboxGroup(labels=["x", "y", "vx","vy","lat","lon"], active=[0, 1])

        # set up plots
        print(datalog)

        datasource = ColumnDataSource(data = dict(time=[],sim=[],real=[]))
        datasource.data = datalog

        realtools = 'xpan,wheel_zoom,xbox_select,reset'
        simtools = 'xpan,wheel_zoom,reset'

        ts1 = figure(plot_width=1200, plot_height=400, tools=realtools, x_axis_type='linear', active_drag="xbox_select")
        ts1.line('time','sim', source=datasource, line_width=2, color="orange", legend_label="Simulated data")
        ts1.line('time','real', source=datasource, line_width=2, color="blue", legend_label="Real data")
        

        x_range_offset = (ulog.last_timestamp - ulog.start_timestamp) * 0.05
        x_range = Range1d(ulog.start_timestamp - x_range_offset, ulog.last_timestamp + x_range_offset)
        flight_mode_changes = get_flight_mode_changes(ulog)

        # set up layout
        widgets = column(datatype,stats)
        sim_button = column(sim_reverse_button)
        real_button = column(real_reverse_button)
        main_row = row(widgets)
        series = column(ts1, sim_button, real_button)
        layout = column(main_row, series)

        # initialize


        update()
        curdoc().add_root(intro_text)

        curdoc().add_root(sim_upload_text)
        curdoc().add_root(file_input)
        curdoc().add_root(real_upload_text)
        curdoc().add_root(file_input2)
        curdoc().add_root(choose_field_text)    
        curdoc().add_root(layout)
        curdoc().title = "Flight data"

        plots = []

    return plots
