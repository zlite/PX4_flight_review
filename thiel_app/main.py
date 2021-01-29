""" This contains Thiel analysis plots """


from os import read, path
# import px4tools
import numpy as np
import math
import io
import os
import sys
import errno
import base64
from db_entry import *
import pickle 

import simstats  # this the module that you can modify to add your own stats

#import thiel_analysis
from bokeh.io import curdoc,output_file, show
from bokeh.models.widgets import Div
from bokeh.models import Title
from bokeh.layouts import column
from scipy.interpolate import interp1d

from plotted_tables import *
from configured_plots import *
from os.path import dirname, join

from config import *
from colors import HTML_color_to_RGB
from helper import *
from leaflet import ulog_to_polyline
from bokeh.models import RadioButtonGroup, Button
from bokeh.models.widgets import Paragraph

import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.application.handlers import DirectoryHandler


#pylint: disable=cell-var-from-loop, undefined-loop-variable,


default_simname = 'sim.ulg'  # these are the defaults if you don't load your own data
default_realname = 'real.ulg'
simdescription = '(Dummy data. Please select your own sim log above)'
realdescription = '(Dummy data. Please select your own real log above)'
sim_polarity = 1  # determines if we should reverse the Y data
real_polarity = 1
simx_offset = 0
realx_offset = 0
read_file = True
get_new_data = True
reverse_sim_data = False
reverse_real_data = False
refresh = False
read_file_local = False
new_real = False
new_sim = False
mission_only = False
sim_metric = 'x'
real_metric = 'x'
tplot_height = 400
tplot_width = 1000
keys = []
labels_text = []
labels_color = []
labels_y_pos = []
labels_x_pos = []
annotations = []
mission_annotations = []
labels = []
sim_label = Label()
real_label = Label()
annotation_counter = 0
mission_annotation_counter = 0
config = [default_simname, default_realname, sim_metric, real_metric, simdescription, realdescription, 1, 1]  # this is just a placeholder in case you don't already have


# def kill():
#     # this is just for debugging. It creates an error so we can watch crash handling
#     sys.exit()
#  #   raise RuntimeError("Fake error")

# kill_mode_button = Button(label="Kill")   # This is just a debugging tool to make sure the web app can handle crashes
# kill_mode_button.on_click(kill)


mission_mode_button = RadioButtonGroup(
        labels=["Show all flight modes", "Show only Mission mode"], active=0)
mission_mode_button.on_change('active', lambda attr, old, new: mission_mode())
normalize_mode_button = RadioButtonGroup(
        labels=["Raw data", "Normalized data"], active=0)
normalize_mode_button.on_change('active', lambda attr, old, new: normalize())
sim_reverse_button = RadioButtonGroup(
        labels=["Sim Default Orientation", "Reversed Orientation"], active=0)
sim_reverse_button.on_change('active', lambda attr, old, new: reverse_sim())
real_reverse_button = RadioButtonGroup(
        labels=["Real Default Orientation", "Reversed Orientation"], active=0)
real_reverse_button.on_change('active', lambda attr, old, new: reverse_real())
sim_swap_button = RadioButtonGroup(
        labels=["Sim Default X/Y", "Swapped X/Y"], active=0)
sim_swap_button.on_change('active', lambda attr, old, new: swap_sim())
real_swap_button = RadioButtonGroup(
        labels=["Real Default X/Y", "Swapped X/Y"], active=0)
real_swap_button.on_change('active', lambda attr, old, new: swap_real())
spacer = Div(text="<hr>", width=800, height=20)
explainer = Div(text="<b>Note:</b> the X/Y coordinate system is set relatively arbitrarily by the drone at startup \
                            and does not reflect GPS positions or compass direction. So you may find that you need to \
                            compare one file's X with another's Y or reverse one to achieve alignment. ", width=800, height=50)
# set up widgets

stats = PreText(text='Thiel Coefficient', width=500)
# datatype = Select(value='XY', options=DEFAULT_FIELDS)
stats2 = PreText(text='Song Coefficient', width=500)


# @lru_cache()
def load_data(filename):
    global keys
    fname = os.path.join(get_log_filepath(), filename)
    if path.exists(fname): 
        ulog = load_ulog_file(fname)
    else:
        print("log does not exist; loading default data instead")
        fname = os.path.join(get_log_filepath(), 'sim.ulg')
        ulog = load_ulog_file(fname) 
    data = ulog.data_list
    for d in data:
        data_keys = [f.field_name for f in d.field_data]
        data_keys.remove('timestamp')
        keys.append(data_keys)
    cur_dataset = ulog.get_dataset('vehicle_local_position')
    flight_mode_changes = get_flight_mode_changes(ulog)
    return cur_dataset, flight_mode_changes


# @lru_cache()
def get_data(simname,realname, sim_metric, real_metric, read_file):
    global dfsim, dfreal, sim_flight_mode_changes, real_flight_mode_changes

    if read_file:
        dfsim, sim_flight_mode_changes = load_data(simname)
        dfreal, real_flight_mode_changes = load_data(realname)
        read_file = False

    sim_data = dfsim.data[sim_metric].copy()  # we copy the data so we can change it wihout changing the original
    sim_time = dfsim.data['timestamp'].copy()
    real_data = dfreal.data[real_metric].copy()
    real_time = dfreal.data['timestamp'].copy()

    if mission_only:                # only show data for when the drone is in auto modes
        temp_pd_sim = pd.DataFrame(sim_data, columns = ['sim'])  # create one dataframe that's just the flight data for the selected metric
        sim_mission_start, sim_mission_end = get_mission_mode(sim_flight_mode_changes)
        pd_sim_time = pd.DataFrame(sim_time,columns = ['time'])
        temp_pd_sim = pd.concat([pd_sim_time,temp_pd_sim], axis=1)
        pd_sim2 = temp_pd_sim.loc[(temp_pd_sim['time'] >= sim_mission_start) & (temp_pd_sim['time'] <= sim_mission_end)]  #slice this just to the mission portion
        pd_sim = pd_sim2.copy()
        starting_sim_time = pd_sim.iat[0,0] 
        pd_sim['time'] -= starting_sim_time  # zero base the time
        pd_sim_time['time'] = pd_sim['time']
        pd_sim = pd_sim.drop(columns=['time'])  # we don't need these old time columns anymore



        temp_pd_real = pd.DataFrame(real_data, columns = ['real'])
        real_mission_start, real_mission_end = get_mission_mode(real_flight_mode_changes)
        pd_real_time = pd.DataFrame(real_time, columns = ['time']) 
        temp_pd_real = pd.concat([pd_real_time,temp_pd_real], axis=1)
#       print("Real mission start, finish", real_mission_start,real_mission_end)
        pd_real2 = temp_pd_real.loc[(temp_pd_real['time'] >= real_mission_start) & (temp_pd_real['time'] <= real_mission_end)] # slice this just to the mission portion
        pd_real = pd_real2.copy()
        starting_real_time = pd_real.iat[0,0]
        pd_real['time'] -= starting_real_time  # zero base the time
        pd_real_time['time'] = pd_real['time']
        pd_real = pd_real.drop(columns=['time'])  # we don't need these old time columns anymore
    
    else:
        pd_sim = pd.DataFrame(sim_data, columns = ['sim'])
        pd_sim_time = pd.DataFrame(sim_time,columns = ['time'])
        starting_sim_time = pd_sim_time.iat[0,0] 
        pd_sim_time['time'] -= starting_sim_time  # zero base the time

        pd_real = pd.DataFrame(real_data, columns = ['real'])
        pd_real_time = pd.DataFrame(real_time, columns = ['time'])
        starting_real_time = pd_real_time.iat[0,0] 
        pd_real_time['time'] -= starting_real_time  # zero base the time


    pd_real_time.dropna(subset=['time'], inplace=True)  # remove empty rows
    pd_real.reset_index(drop=True, inplace=True)   # reset all the indicies to zero
    pd_real_time.reset_index(drop=True, inplace=True)
    pd_sim_time.dropna(subset=['time'], inplace=True) # do the same for the sims
    pd_sim.reset_index(drop=True, inplace=True)
    pd_sim_time.reset_index(drop=True, inplace=True)


    if len(pd_sim_time) > len(pd_real_time):  # base the y axis on the longest time
        pd_time = pd_sim_time
    else:
        pd_time = pd_real_time
    

    new_data = pd.concat([pd_time,pd_sim, pd_real], axis=1)

    save_settings(config)
    return new_data

def update_config():
    config[0] = simname
    config[1] = realname
    config[2] = sim_metric
    config[3] = real_metric
    config[4] = simdescription
    config[5] = realdescription
    config[6] = 0
    config[7] = 0
    return config

def save_settings(config):
    with open('settings', 'wb') as fp:  #save state
        pickle.dump(config, fp)

def read_settings():
    ''' We're now going to load a bunch of state variables to sync the app back to the last known state. The file "settings" should exist in the main directory

        # config = [simname, realname, sim_metric, real_metric, simdescription, realdescription]

        The format of the list is as follows:
        config[0] = sim ID
        config[1] = real ID
        config[2] = sim_metric
        config[3] = real_metric
        config[4] = simdescription
        config[5] = realdesciption
        config[6] = real_reverse_button.active
        config[7] = sim_reverse_button.active

        '''
    global simname, realname, sim_metric, real_metric, simdescription, realdescription, real_reverse_button, sim_reverse_button

    if path.exists('settings'):    
        with open ('settings', 'rb') as fp:
            config = pickle.load(fp)
        simname = config[0]
        realname = config[1]
        sim_metric = config[2]        
        real_metric = config[3]
        simdescription = str(config[4])
        realdescription = str(config[5])
        # real_reverse_button.active = config[5]
        # sim_reverse_button.active = config[6]
    else:   # the app is running for the first time, so start with dummy data
        simname = "sim.ulg"
        realname = "real.ulg"
        sim_metric = 'x'
        real_metric = 'x'
        simdescription = "Dummy simulation data"
        realdescription = "Dummy real data"
        config = update_config()
        print("Starting with dummy data", config)
    return config

def get_mission_mode(flight_mode_changes):
    # time_offset, null = flight_mode_changes[0]  # zero base the time
    m_start = 0
    m_end = 0
    for i in range(len(flight_mode_changes)-1):
            t_start, mode = flight_mode_changes[i]
#            t_start = t_start - time_offset
            t_end, mode_next = flight_mode_changes[i + 1]
#            t_end = t_end - time_offset
            if mode in flight_modes_table:
                mode_name, color = flight_modes_table[mode]
                if mode_name == 'Mission':
                    m_start = int(t_start)
                    m_end = int(t_end)
    return m_start, m_end


def plot_flight_modes(flight_mode_changes,type):
    global annotations, mission_annotations, annotation_counter, mission_annotation_counter, sim_label, real_label, labels, ts1

    if mission_only:
        for i in range(annotation_counter):
            annotations[i].visible = False  # turn off the previous annotations

        for j in range(len(labels)): # Turn off the previous labels
            labels[j].visible = False

        if type == 'sim':
            real_label.visible = True # now just turn on the two mission mode labels
        else:
            sim_label.visible = True 


    labels_y_pos = []
    labels_x_pos = []
    labels_text = []
    labels_color = []
    added_box_annotation_args = {}
    if type == 'sim':
        labels_y_offset = tplot_height - 300      # plot the sim shaded areas below the real ones
    else:
        labels_y_offset = tplot_height - 200

    time_offset, null = flight_mode_changes[0]  # zero base the time
    for i in range(len(flight_mode_changes)-1):
        t_start, mode = flight_mode_changes[i]
        t_start = t_start - time_offset
        t_end, mode_next = flight_mode_changes[i + 1]
        t_end = t_end - time_offset
        if mode in flight_modes_table:
            mode_name, color = flight_modes_table[mode]
            if mission_only:
                if mode_name == 'Mission':
                    mtime_offset = t_start
                    mt_start = 0   # zero base mission mode
                    mt_end = t_end - mtime_offset
                    annotation = BoxAnnotation(left=int(mt_start), right=int(mt_end), top = labels_y_offset, bottom = labels_y_offset-100, 
                                        fill_alpha=0.09, line_color='black', top_units = 'screen',bottom_units = 'screen',
                                        fill_color=color, **added_box_annotation_args)
                    annotation.visible = True
                    mission_annotations.append(annotation)   # add the box to the list of annotations, so we can remove it if necessary later
                    mission_annotation_counter = mission_annotation_counter + 1  # increment the list of annotations
                    ts1.add_layout(annotation)
            else:
                annotation = BoxAnnotation(left=int(t_start), right=int(t_end), top = labels_y_offset, bottom = labels_y_offset-100, 
                                    fill_alpha=0.09, line_color='black', top_units = 'screen',bottom_units = 'screen',
                                    fill_color=color, **added_box_annotation_args)
                annotation.visible = True
                annotations.append(annotation)   # add the box to the list of annotations, so we can remove it if necessary later
                annotation_counter = annotation_counter + 1  # increment the list of annotations
                ts1.add_layout(annotation)



            if flight_mode_changes[i+1][0] - t_start > 1e6: # filter fast
                                                 # switches to avoid overlap
                if type == 'sim':
                    labels_text.append(mode_name)
                else:
                    labels_text.append(mode_name)

                labels_x_pos.append(t_start)
                labels_y_pos.append(labels_y_offset)
                labels_color.append(color)

        # plot flight mode names as labels

            if len(labels_text) > 0:
                source = ColumnDataSource(data=dict(x=labels_x_pos, text=labels_text,
                                                    y=labels_y_pos, textcolor=labels_color))
                if type == 'sim':
                    label_color = 'orange'
                else:
                    label_color = 'blue'
                if mission_only:
                    if mode_name == 'Mission': 
                        label = Label(x=t_start, y=labels_y_offset, text='Mission',  # just create a single label for each mission mode
                                            y_units='screen', level='underlay',
                                            render_mode='canvas',
                                            text_font_size='10pt',
                                            text_color= label_color, text_alpha=0.85,
                                            background_fill_color='white',
                                            background_fill_alpha=0.8, angle=90, angle_units = 'deg', text_align='right', text_baseline='top')

                        if type == 'sim':
                            sim_label = label
                        else:
                            real_label = label
                        ts1.add_layout(label)
                else:
                    label = LabelSet(x='x', y='y', text='text',   # create a whole label set
                                    y_units='screen', level='underlay',
                                    source=source, render_mode='canvas',
                                    text_font_size='10pt',
                                    text_color= label_color, text_alpha=0.85,
                                    background_fill_color='white',
                                    background_fill_alpha=0.8, angle=90/180*np.pi,
                                    text_align='right', text_baseline='top')
                    labels.append(label)
                    ts1.add_layout(label)

def update(selected=None):
    global reverse_sim_data, reverse_real_data, datalog, original_data, datasource, ts1, get_new_data
    clear_boxes() #turn off old mode displays
    if get_new_data: 
        print("Fetching new data", simname, realname, sim_metric, real_metric, read_file)
        original_data = get_data(simname, realname, sim_metric, real_metric, read_file)
        datalog = copy.deepcopy(original_data)
        get_new_data = False
 
    if reverse_sim_data:
        datalog[['sim']] = sim_polarity * original_data['sim']  # reverse data if necessary
        reverse_sim_data = False
    if reverse_real_data:
        datalog['real'] = real_polarity * original_data['real']
        reverse_real_data = False  

    datasource.data = datalog
    plot_flight_modes(sim_flight_mode_changes, 'sim')
    plot_flight_modes(real_flight_mode_changes, 'real')

    config = update_config()
    thiel = simstats.sim2real_stats(datalog)
    song = simstats.sim2real_stats2(datalog)
    trend = simstats.equation_8(datalog[['sim']],datalog[['real']])
    print("trend= ", trend)

    stats.text = 'Thiel coefficient (1 = no correlation, 0 = perfect): ' + str(thiel)
    stats2.text = 'Song coefficient (0 = perfect): ' + str(song)
    save_settings(config)

def normalize():
    global datalog, realnorm, simnorm, get_new_data, norm
    if (normalize_mode_button.active == 1):
        norm = True
        realnorm = 0
        simnorm = 0 
        sim_mean = datalog['sim'].mean()  # get the average
        real_mean = datalog['real'].mean()
        if sim_mean >= real_mean:
            realnorm = sim_mean - real_mean
            datalog['real'] = datalog['real'] + realnorm  # increase the lower one by the average of their difference
        else:
            simnorm = real_mean - sim_mean
            datalog['sim'] = datalog['sim'] + simnorm
    else:
        norm = False
        datalog['real'] = datalog['real'] - realnorm  # revert to the way they were
        datalog['sim'] = datalog['sim'] - simnorm
    get_new_data = False
    update()

def clear_boxes():
    global annotations, mission_annotations
    for i in range(mission_annotation_counter):
        mission_annotations[i].visible = False  # turn off the previous mission annotations
    for j in range(annotation_counter):
        annotations[j].visible = False  # turn off the previous other mode annotations

def mission_mode():
    global mission_only, get_new_data
    if (mission_mode_button.active == 1):   
        mission_only = True
        print("Show only missions")
    else: 
        mission_only = False
        print("Show all modes")
    get_new_data = True
    normalize_mode_button.active = 0
    update()

def reverse_sim():
    global sim_polarity, reverse_sim_data, config
    if (sim_reverse_button.active == 1): 
        sim_polarity = -1
        config[6] = sim_reverse_button.active
    else: sim_polarity = 1
    reverse_sim_data = True
    normalize_mode_button.active = 0
    update()

def reverse_real():
    global real_polarity, reverse_real_data, config
    if (real_reverse_button.active == 1): 
        real_polarity = -1
        config[5] = real_reverse_button.active
    else: real_polarity = 1
    reverse_real_data = True
    normalize_mode_button.active = 0
    update()

def swap_sim():
    global sim_metric, get_new_data
    print("Swapping sim. Metric is", sim_metric)
    if sim_metric == 'x': 
        sim_metric = 'y'
    else: 
        sim_metric = 'x'
    get_new_data = True
    normalize_mode_button.active = 0
    update()

def swap_real():
    global real_metric, get_new_data
    print("Swapping real. Metric is", real_metric)
    if real_metric == 'x': 
        real_metric = 'y'
    else:
        real_metric = 'x'
    get_new_data = True
    normalize_mode_button.active = 0
    update()

def sim_change(attrname, old, new):
    global sim_metric, real_metric, read_file, config, get_new_data
    print("Sim change:", new)
    sim_metric = new
    real_metric = new
    config[2] = sim_metric # save state
    config[3] = real_metric # save state
    get_new_data = True
    read_file = True
    normalize_mode_button.active = 0
    update()   

def get_thiel_analysis_plots(simname, realname):
    global datalog, original_data, datasource, layout, ts1, chart, annotation_counter

    additional_links= "<b><a href='/browse?search=sim'>Load Simulation Log</a> <p> <a href='/browse?search=real'>Load Real Log</a></b>" 
    save_settings(config)
    datalog = get_data(simname, realname, sim_metric, real_metric, read_file)
    original_data = copy.deepcopy(datalog)

    for i in range(10):
        if keys[i][0] == 'x':
            found_x = i
    
    datatype = Select(value='x', options=keys[found_x])

    datatype.on_change('value', sim_change)

    intro_text = Div(text="""<H2>Sim/Real Thiel Coefficient Calculator</H2> \
        <p> Load two PX4 datalogs, one a real flight and the other a simulation of that flight, \
            and see how well they compare. We use the well-known <a href="https://www.vosesoftware.com/riskwiki/Thielinequalitycoefficient.php">Thiel Coefficient</a> and <a href="https://drive.google.com/file/d/1XY8aZz89emFt-LAuUZ2pjC1GHwRARr9f/view">Song variation</a> of that to generate correspondence scores.""",width=800, height=100, align="center")
    choose_field_text = Paragraph(text="Choose a data field to compare:",width=500, height=15)
    links_text = Div(text="<table width='100%'><tr><td><h3>" + "</h3></td><td align='left'>" + additional_links+"</td></tr></table>")
    datasource = ColumnDataSource(data = dict(time=[],sim=[],real=[]))
    datasource.data = datalog

    tools = 'xpan,wheel_zoom,reset'
    ts1 = figure(plot_width=tplot_width, plot_height=tplot_height, tools=tools, x_axis_type='linear')
 
  #  ts1.add_layout(Legend(), 'right')    # if you want the legend outside of the plot
    print("real description", realdescription)
    ts1.line('time','sim', source=datasource, line_width=2, color="orange", legend_label="Simulated data: "+ simdescription)
    ts1.line('time','real', source=datasource, line_width=2, color="blue", legend_label="Real data: " + realdescription)
    ts1.legend.background_fill_alpha = 0.7   # make the background of the legend more transparent

    ts1.add_layout(Title(text="Time (seconds)", align="center"), "below")
 #   annotation_counter = annotation_counter + 1  # increment the list of annotations
    # x_range_offset = (datalog.last_timestamp - datalog.start_timestamp) * 0.05
    # x_range = Range1d(datalog.start_timestamp - x_range_offset, datalog.last_timestamp + x_range_offset)


    plot_flight_modes(sim_flight_mode_changes, 'sim')
    plot_flight_modes(real_flight_mode_changes, 'real')
    



    # set up layout
    widgets = column(datatype,stats,stats2)
    mission_button = column(mission_mode_button)
    normalize_button = column(normalize_mode_button)
    sim_button = column(sim_reverse_button)
    real_button = column(real_reverse_button)
    sswap_button = column(sim_swap_button)
    rswap_button = column(real_swap_button)
    rule = column(explainer)
    space = column(spacer)
    main_row = row(widgets)
    chart = column(ts1)
    buttons = column(mission_button, normalize_button, space, sim_button, sswap_button, rule, real_button, rswap_button)
    layout = column(main_row, chart, buttons)

    # initialize

    update()
    curdoc().add_root(intro_text)
    curdoc().add_root(links_text)
    curdoc().add_root(choose_field_text)    
    curdoc().add_root(layout)
    curdoc().title = "Flight data"



print("Now starting Thiel app")
GET_arguments = curdoc().session_context.request.arguments
config = read_settings()
print("simname is", simname, "realname is", realname)


if GET_arguments is not None and 'log' in GET_arguments:
    log_args = GET_arguments['log']
    if len(log_args) == 1:
        templog_id = str(log_args[0], 'utf-8')
        file_details = templog_id.split('desc:')
        templog_id = file_details[0]
        if (templog_id.find("sim") != -1):
            log_id = templog_id.replace('sim','')
            print("This is a sim file. New log ID=", log_id)
            ulog_file_name = get_log_filename(log_id)
            simname = os.path.join(get_log_filepath(), ulog_file_name)
            simdescription = str(file_details[1])
        elif (templog_id.find("real") != -1):
            log_id = templog_id.replace('real','')
            print("This is a real file. New log ID=", log_id)
            ulog_file_name = get_log_filename(log_id)
            realname = os.path.join(get_log_filepath(), ulog_file_name)
            realdescription = str(file_details[1])
        else:
            if not validate_log_id(templog_id):
                raise ValueError('Invalid log id: {}'.format(log_id))
        print('GET[log]={}'.format(templog_id))
        ulog_file_name = get_log_filename(templog_id)
get_thiel_analysis_plots(simname, realname)