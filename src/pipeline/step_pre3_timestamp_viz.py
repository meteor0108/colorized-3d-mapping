# https://stackoverflow.com/questions/76931490/plotly-adding-a-timeline-using-add-trace-to-go-figure-object

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px




# data = {'index': index_list, 'topic': topic_list, 'rosbag_time': rosbag_time_list, 'header_time': header_time_list}

import argparse
from src.common import load_config

_parser = argparse.ArgumentParser(description="step_pre3: timestamp Gantt chart (plotly)")
_parser.add_argument("--config", default="default.yaml")
_parser.add_argument("--csv", default=None, help="*_timestamp.csv (없으면 timestamp_viz.csv_path)")
_args, _ = _parser.parse_known_args()
_cfg = load_config(_args.config)
csv_path = _args.csv or _cfg.timestamp_viz.csv_path
if not csv_path:
    raise ValueError("timestamp_viz.csv_path가 비어있고 --csv도 지정되지 않았습니다.")
df = pd.read_csv(csv_path)

df['Start_Time'] = df['header_time']
df['End_Time'] = df['header_time']+3600




# mydata_as_json = '{"Start_Time":{"0":"2023-11-01 16:48:24.8", \
# "1":"2023-11-01 16:49:47.7",\
# "2":"2023-11-01 16:53:23.3",\
# "3":"2023-11-01 16:56:08.6",\
# "4":"2023-11-01 16:58:37.9",\
# "5":"2023-11-01 17:00:10.9"},"End_Time":{"0":"2023-11-01 16:48:37.7","1":"2023-11-01 16:51:44.3","2":"2023-11-01 16:54:00.2","3":"2023-11-01 16:57:23.4","4":"2023-11-01 16:58:56.8","5":"2023-11-01 17:01:59.1"},"Instr":{"0":"MVIC","1":"LEISA","2":"MVIC","3":"LEISA","4":"MVIC","5":"LEISA"}}'

# df = pd.DataFrame(eval(mydata_as_json))
print(df)

fig_bad = go.Figure()
px_timeline = px.timeline(
            df, 
            x_start='Start_Time', 
            x_end='End_Time', 
            y='topic',
        )
fig_bad = fig_bad.add_trace(
        px_timeline.data[0]
)
fig_bad.layout = px_timeline.layout
fig_bad.show()

# print(df['index'])
# for i in range(df.shape[0]):

