from __future__ import print_function
import px4tools
import numpy as np
import math
import io
import os
import sys
import errno
from plotting import DataPlot
from functools import lru_cache
from os.path import dirname, join
from bokeh.io import output_file, show
from bokeh.models.widgets import FileInput
from bokeh.models.widgets import Paragraph
from bokeh.models import CheckboxGroup
from bokeh.models import RadioButtonGroup
from bokeh.models import Range1d
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.application.handlers import DirectoryHandler


import time
import copy
from bokeh.models import Div


import pandas as pd
import argparse

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure

DATA_DIR = join(dirname(__file__), 'datalogs')

DEFAULT_FIELDS = ['XY', 'LatLon', 'VxVy']

simname = 'airtonomysim.csv'
realname = 'airtonomyreal.csv'
sim_polarity = 1  # determines if we should reverse the Y data
real_polarity = 1
simx_offset = 0
realx_offset = 0
read_file = True
reverse_sim_data = False
reverse_real_data = False
new_data = True


"""
Tornado handler for the upload page
"""


import datetime
import os
from html import escape
import sys
import uuid
import binascii
import sqlite3
import tornado.web
from tornado.ioloop import IOLoop

from pyulog import ULog
from pyulog.px4 import PX4ULog

# this is needed for the following imports
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../plot_app'))
from db_entry import DBVehicleData, DBData
from config import get_db_filename, get_http_protocol, get_domain_name, \
    email_notifications_config
from helper import get_total_flight_time, validate_url, get_log_filename, \
    load_ulog_file, get_airframe_name, ULogException
from overview_generator import generate_overview_img_from_id

#pylint: disable=relative-beyond-top-level
from .common import get_jinja_env, CustomHTTPError, generate_db_data_from_log_file, \
    TornadoRequestHandlerBase
from .send_email import send_notification_email, send_flightreport_email
from .multipart_streamer import MultiPartStreamer


UPLOAD_TEMPLATE = 'thiel.html'


#pylint: disable=attribute-defined-outside-init,too-many-statements, unused-argument

@lru_cache()
def load_data_sim(simname):
    fname = join(DATA_DIR, simname)
    data = pd.read_csv(fname)
    dfsim = pd.DataFrame(data)
    return dfsim

@lru_cache()
def load_data_real(realname):
    fname = join(DATA_DIR, realname)
    data = pd.read_csv(fname)
#   select_data.to_numpy()  # convert to a numpy array
    dfreal = pd.DataFrame(data)
    return dfreal


@lru_cache()
def get_data(simname,realname):
    global select_data
    dfsim = load_data_sim(simname)
    dfreal = load_data_real(realname)
    data = pd.concat([dfsim, dfreal], axis=1)
    data = data.dropna()   # remove missing values
    sim_mean = data.simy.mean()  # get the average
    real_mean = data.realy.mean()
    mean_diff = sim_mean - real_mean 
    data.realy = data.realy + mean_diff # normalize the two
    data['simy'] = data.simy
    data['simx'] = data.simx
    data['realy'] = data.realy
    data['realx'] = data.realx
    select_data=np.asarray(data)  # convert to an array for real selection line
#    original_data = copy.deepcopy(data)
    return data

# set up widgets

stats = PreText(text='Thiel Coefficient', width=500)
datatype = Select(value='XY', options=DEFAULT_FIELDS)

# set up plots

simsource = ColumnDataSource(data = dict(simx=[],simy=[]))
simsource_static = ColumnDataSource(data = dict(simx=[],simy=[]))
realsource = ColumnDataSource(data = dict(realx=[],realy=[]))
realsource_static = ColumnDataSource(data = dict(realx=[],realy=[]))

realtools = 'xpan,wheel_zoom,xbox_select,reset'
simtools = 'xpan,wheel_zoom,reset'

ts1 = figure(plot_width=900, plot_height=200, tools=realtools, x_axis_type='linear', active_drag="xbox_select")
ts1.line('simx', 'simy', source=simsource, line_width=2)
ts1.circle('simx', 'simy', size=1, source=simsource_static, color=None, selection_color="orange")

ts2 = figure(plot_width=900, plot_height=200, tools=simtools, x_axis_type='linear')
# to adjust ranges, add something like this: x_range=Range1d(0, 1000), y_range = None,
# ts2.x_range = ts1.x_range
ts2.line('realx', 'realy', source=realsource, line_width=2)
ts2.circle('realx', 'realy', size=1, source=realsource_static, color="orange")

# set up callbacks

def sim_change(attrname, old, new):
    real.options = nix(new, DEFAULT_FIELDS)
    update()

def update(selected=None):
    global read_file, reverse_sim_data, reverse_real_data, new_data, simsource, simsource_static, realsource, realsource_static,original_data, data, data_static, new_data, select_data, select_datadf
    if (read_file):
        original_data = get_data(simname, realname)
        data = copy.deepcopy(original_data)
        data_static = copy.deepcopy(original_data)
        read_file = False
    print("Sim offset", simx_offset)
    print("Real offset", realx_offset)
    if reverse_sim_data:
        data[['simy']] = sim_polarity * original_data[['simy']]  # reverse data if necessary
        data_static[['simy']] = sim_polarity * original_data[['simy']]  # reverse data if necessary
        simsource.data = data
        simsource_static.data = data_static
        simmax = round(max(data[['simy']].values)[0])  # reset the axis scales as appopriate (auto scaling doesn't work)
        simmin = round(min(data[['simy']].values)[0])
        ts1.y_range.start = simmin - abs((simmax-simmin)/10)
        ts1.y_range.end = simmax + abs((simmax-simmin)/10)
        reverse_sim_data = False
    if reverse_real_data:
        data[['realy']] = real_polarity * original_data[['realy']]
        data_static[['realy']] = real_polarity * original_data[['realy']]
        realsource.data = data
        select_datadf[['realy']] = -1 * select_datadf[['realy']]
        realsource_static.data = select_datadf
        realmax = round(max(data[['realy']].values)[0])
        realmin = round(min(data[['realy']].values)[0])
        ts2.y_range.start = realmin - abs((realmax-realmin)/10)
        ts2.y_range.end = realmax + abs((realmax-realmin)/10)
        reverse_real_data = False
    if new_data:
        simsource.data = data[['simx', 'simy','realx','realy']]
        simsource_static.data = data_static[['simx', 'simy','realx','realy']]
        realsource.data = data[['simx', 'simy','realx','realy']]
        select_datadf = pd.DataFrame({'realx': select_data[:, 2], 'realy': select_data[:, 3]})  # convert back to a pandas dataframe     
        realsource_static.data = select_datadf
        new_data = False
#    select_data = copy.deepcopy(tempdata)
    ts1.title.text, ts2.title.text = 'Sim', 'Real'

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

datatype.on_change('value', sim_change)

def simselection_change(attrname, old, new):
    global data_static, new_data, realx_offset, realsource_static,select_data
    selected = simsource_static.selected.indices
    if selected:
        seldata = data.iloc[selected, :]
        sorted_data = seldata.sort_values(by=['simx'])
        start = int(sorted_data.values[0][0])
        print("Start =", start)
    if (len(seldata['simx']) != 0):
        for x in range(len(select_data)):
            select_data[x][2] = 0    #zero out the data
            select_data[x][3] = 0
        for x in range(start, (start+len(sorted_data['simx'])-1)):
            tempx = int(sorted_data['realx'][x] + realx_offset - simx_offset)
            select_data[tempx][2] = realsource.data['realx'][tempx]
            select_data[tempx][3] = realsource.data['realy'][tempx]
        update_stats(seldata)
    new_data = True
    update()

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


def startserver(doc):
    file_input = FileInput(accept=".ulg, .csv")
    file_input.on_change('value', upload_new_data_sim)
    file_input2 = FileInput(accept=".ulg, .csv")
    file_input2.on_change('value', upload_new_data_real)

    intro_text = Div(text="""<H2>Sim/Real Thiel Coefficient Calculator</H2>""",width=500, height=100, align="center")
    sim_upload_text = Paragraph(text="Upload a simulator datalog:",width=500, height=15)
    real_upload_text = Paragraph(text="Upload a corresponding real-world datalog:",width=500, height=15)
    #checkbox_group = CheckboxGroup(labels=["x", "y", "vx","vy","lat","lon"], active=[0, 1])

    sim_reverse_button = RadioButtonGroup(
            labels=["Sim Default", "Reversed"], active=0)
    sim_reverse_button.on_change('active', lambda attr, old, new: reverse_sim())
    real_reverse_button = RadioButtonGroup(
            labels=["Real Default", "Reversed"], active=0)
    real_reverse_button.on_change('active', lambda attr, old, new: reverse_real())

    simsource_static.selected.on_change('indices', simselection_change)


    # The below are in case you want to see the x axis range change as you pan. Poorly documented elsewhere!
    #ts1.x_range.on_change('end', lambda attr, old, new: print ("TS1 X range = ", ts1.x_range.start, ts1.x_range.end))
    #ts2.x_range.on_change('end', lambda attr, old, new: print ("TS2 X range = ", ts2.x_range.start, ts2.x_range.end))

    ts1.x_range.on_change('end', lambda attr, old, new: change_sim_scale(ts1.x_range.start))
    ts2.x_range.on_change('end', lambda attr, old, new: change_real_scale(ts2.x_range.start))

    # set up layout
    widgets = column(datatype,stats)
    sim_button = column(sim_reverse_button)
    real_button = column(real_reverse_button)
    main_row = row(widgets)
    series = column(ts1, sim_button, ts2, real_button)
    layout = column(main_row, series)

    # initialize
    update()
    doc.add_root(intro_text)

    doc.add_root(sim_upload_text)
    doc.add_root(file_input)
    doc.add_root(real_upload_text)
    doc.add_root(file_input2)
    doc.add_root(layout)
    doc.title = "Flight data"




def update_vehicle_db_entry(cur, ulog, log_id, vehicle_name):
    """
    Update the Vehicle DB entry
    :param cur: DB cursor
    :param ulog: ULog object
    :param vehicle_name: new vehicle name or '' if not updated
    :return vehicle_data: DBVehicleData object
    """

    vehicle_data = DBVehicleData()
    if 'sys_uuid' in ulog.msg_info_dict:
        vehicle_data.uuid = escape(ulog.msg_info_dict['sys_uuid'])

        if vehicle_name == '':
            cur.execute('select Name '
                        'from Vehicle where UUID = ?', [vehicle_data.uuid])
            db_tuple = cur.fetchone()
            if db_tuple is not None:
                vehicle_data.name = db_tuple[0]
            print('reading vehicle name from db:'+vehicle_data.name)
        else:
            vehicle_data.name = vehicle_name
            print('vehicle name from uploader:'+vehicle_data.name)

        vehicle_data.log_id = log_id
        flight_time = get_total_flight_time(ulog)
        if flight_time is not None:
            vehicle_data.flight_time = flight_time

        # update or insert the DB entry
        cur.execute('insert or replace into Vehicle (UUID, LatestLogId, Name, FlightTime)'
                    'values (?, ?, ?, ?)',
                    [vehicle_data.uuid, vehicle_data.log_id, vehicle_data.name,
                     vehicle_data.flight_time])
    return vehicle_data


@tornado.web.stream_request_body
class ThielHandler(TornadoRequestHandlerBase):
    """ Upload log file Tornado request handler: handles page requests and POST
    data """

    def initialize(self):
        """ initialize the instance """
        self.multipart_streamer = None

    def prepare(self):
        """ called before a new request """
        if self.request.method.upper() == 'POST':
            if 'expected_size' in self.request.arguments:
                self.request.connection.set_max_body_size(
                    int(self.get_argument('expected_size')))
            try:
                total = int(self.request.headers.get("Content-Length", "0"))
            except KeyError:
                total = 0
            self.multipart_streamer = MultiPartStreamer(total)

    def data_received(self, chunk):
        """ called whenever new data is received """
        if self.multipart_streamer:
            self.multipart_streamer.data_received(chunk)

    def get(self, *args, **kwargs):
        """ GET request callback """
        template = get_jinja_env().get_template(UPLOAD_TEMPLATE)
        self.write(template.render())

    def post(self, *args, **kwargs):
        """ POST request callback """
        if self.multipart_streamer:
            try:
                self.multipart_streamer.data_complete()
                form_data = self.multipart_streamer.get_values(
                    ['description', 'email',
                     'allowForAnalysis', 'obfuscated', 'source', 'type',
                     'feedback', 'windSpeed', 'rating', 'videoUrl', 'public',
                     'vehicleName'])
                description = escape(form_data['description'].decode("utf-8"))
                email = form_data['email'].decode("utf-8")
                upload_type = 'personal'
                if 'type' in form_data:
                    upload_type = form_data['type'].decode("utf-8")
                source = 'webui'
                title = '' # may be used in future...
                if 'source' in form_data:
                    source = form_data['source'].decode("utf-8")
                obfuscated = 0
                if 'obfuscated' in form_data:
                    if form_data['obfuscated'].decode("utf-8") == 'true':
                        obfuscated = 1
                allow_for_analysis = 0
                if 'allowForAnalysis' in form_data:
                    if form_data['allowForAnalysis'].decode("utf-8") == 'true':
                        allow_for_analysis = 1
                feedback = ''
                if 'feedback' in form_data:
                    feedback = escape(form_data['feedback'].decode("utf-8"))
                wind_speed = -1
                rating = ''
                stored_email = ''
                video_url = ''
                is_public = 0
                vehicle_name = ''
                error_labels = ''

                if upload_type == 'flightreport':
                    try:
                        wind_speed = int(escape(form_data['windSpeed'].decode("utf-8")))
                    except ValueError:
                        wind_speed = -1
                    rating = escape(form_data['rating'].decode("utf-8"))
                    if rating == 'notset': rating = ''
                    stored_email = email
                    # get video url & check if valid
                    video_url = escape(form_data['videoUrl'].decode("utf-8"), quote=True)
                    if not validate_url(video_url):
                        video_url = ''
                    if 'vehicleName' in form_data:
                        vehicle_name = escape(form_data['vehicleName'].decode("utf-8"))

                    # always allow for statistical analysis
                    allow_for_analysis = 1
                    if 'public' in form_data:
                        if form_data['public'].decode("utf-8") == 'true':
                            is_public = 1

                file_obj = self.multipart_streamer.get_parts_by_name('filearg')[0]
                upload_file_name = file_obj.get_filename()

                while True:
                    log_id = str(uuid.uuid4())
                    new_file_name = get_log_filename(log_id)
                    if not os.path.exists(new_file_name):
                        break

                # read file header & check if really an ULog file
                header_len = len(ULog.HEADER_BYTES)
                if (file_obj.get_payload_partial(header_len) !=
                        ULog.HEADER_BYTES):
                    if upload_file_name[-7:].lower() == '.px4log':
                        raise CustomHTTPError(
                            400,
                            'Invalid File. This seems to be a px4log file. '
                            'Upload it to <a href="http://logs.uaventure.com" '
                            'target="_blank">logs.uaventure.com</a>.')
                    raise CustomHTTPError(400, 'Invalid File')

                print('Moving uploaded file to', new_file_name)
                file_obj.move(new_file_name)

                if obfuscated == 1:
                    # TODO: randomize gps data, ...
                    pass

                # generate a token: secure random string (url-safe)
                token = str(binascii.hexlify(os.urandom(16)), 'ascii')

                # Load the ulog file but only if not uploaded via CI.
                # Then we open the DB connection.
                ulog = None
                if source != 'CI':
                    ulog_file_name = get_log_filename(log_id)
                    ulog = load_ulog_file(ulog_file_name)


                # put additional data into a DB
                con = sqlite3.connect(get_db_filename())
                cur = con.cursor()
                cur.execute(
                    'insert into Logs (Id, Title, Description, '
                    'OriginalFilename, Date, AllowForAnalysis, Obfuscated, '
                    'Source, Email, WindSpeed, Rating, Feedback, Type, '
                    'videoUrl, ErrorLabels, Public, Token) values '
                    '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [log_id, title, description, upload_file_name,
                     datetime.datetime.now(), allow_for_analysis,
                     obfuscated, source, stored_email, wind_speed, rating,
                     feedback, upload_type, video_url, error_labels, is_public, token])

                if ulog is not None:
                    vehicle_data = update_vehicle_db_entry(cur, ulog, log_id, vehicle_name)
                    vehicle_name = vehicle_data.name

                con.commit()

                url = '/plot_app?log='+log_id
                full_plot_url = get_http_protocol()+'://'+get_domain_name()+url
                print(full_plot_url)

                delete_url = get_http_protocol()+'://'+get_domain_name()+ \
                    '/edit_entry?action=delete&log='+log_id+'&token='+token

                # information for the notification email
                info = {}
                info['description'] = description
                info['feedback'] = feedback
                info['upload_filename'] = upload_file_name
                info['type'] = ''
                info['airframe'] = ''
                info['hardware'] = ''
                info['uuid'] = ''
                info['software'] = ''
                info['rating'] = rating
                if len(vehicle_name) > 0:
                    info['vehicle_name'] = vehicle_name

                if ulog is not None:
                    px4_ulog = PX4ULog(ulog)
                    info['type'] = px4_ulog.get_mav_type()
                    airframe_name_tuple = get_airframe_name(ulog)
                    if airframe_name_tuple is not None:
                        airframe_name, airframe_id = airframe_name_tuple
                        if len(airframe_name) == 0:
                            info['airframe'] = airframe_id
                        else:
                            info['airframe'] = airframe_name
                    sys_hardware = ''
                    if 'ver_hw' in ulog.msg_info_dict:
                        sys_hardware = escape(ulog.msg_info_dict['ver_hw'])
                        info['hardware'] = sys_hardware
                    if 'sys_uuid' in ulog.msg_info_dict and sys_hardware != 'SITL':
                        info['uuid'] = escape(ulog.msg_info_dict['sys_uuid'])
                    branch_info = ''
                    if 'ver_sw_branch' in ulog.msg_info_dict:
                        branch_info = ' (branch: '+ulog.msg_info_dict['ver_sw_branch']+')'
                    if 'ver_sw' in ulog.msg_info_dict:
                        ver_sw = escape(ulog.msg_info_dict['ver_sw'])
                        info['software'] = ver_sw + branch_info


                if upload_type == 'flightreport' and is_public and source != 'CI':
                    destinations = set(email_notifications_config['public_flightreport'])
                    if rating in ['unsatisfactory', 'crash_sw_hw', 'crash_pilot']:
                        destinations = destinations | \
                            set(email_notifications_config['public_flightreport_bad'])
                    send_flightreport_email(
                        list(destinations),
                        full_plot_url,
                        DBData.rating_str_static(rating),
                        DBData.wind_speed_str_static(wind_speed), delete_url,
                        stored_email, info)

                    # also generate the additional DB entry
                    # (we may have the log already loaded in 'ulog', however the
                    # lru cache will make it very quick to load it again)
                    generate_db_data_from_log_file(log_id, con)
                    # also generate the preview image
                    IOLoop.instance().add_callback(generate_overview_img_from_id, log_id)

                con.commit()
                cur.close()
                con.close()

                # send notification emails
                send_notification_email(email, full_plot_url, delete_url, info)

                # do not redirect for QGC
                if source != 'QGroundControl':
                    self.redirect(url)

            except CustomHTTPError:
                raise

            except ULogException:
                raise CustomHTTPError(
                    400,
                    'Failed to parse the file. It is most likely corrupt.')
            except:
                print('Error when handling POST data', sys.exc_info()[0],
                      sys.exc_info()[1])
                raise CustomHTTPError(500)

            finally:
                self.multipart_streamer.release_parts()
