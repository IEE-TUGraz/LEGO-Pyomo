import pandas as pd
import os
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, RangeTool, HoverTool, DatePicker, TextInput, Button
from bokeh.layouts import column, row
import datetime as dt
from bokeh.io import curdoc
from bokeh.layouts import column, row


# load some data for rings
path_results = os.path.join("data", "rings_base_example", "results")
scenario = "results_intv15"

# load results excel
df_vGenInvest = pd.read_excel(os.path.join(path_results, f"{scenario}.xlsx"), sheet_name="vGenInvest")

# load vGenP data
df_vGenP = pd.read_excel(os.path.join(path_results, f"{scenario}.xlsx"), sheet_name="vGenP")

# load vCurtailment data
df_vCurtailment = pd.read_excel(os.path.join(path_results, f"{scenario}.xlsx"), sheet_name="vCurtailment")

# load vImpExp data
df_vImpExp = pd.read_excel(os.path.join(path_results, f"{scenario}.xlsx"), sheet_name="vImpExp")

# create a datatime
time_steps = len(df_vGenP.index)
# calculate the time span in hours
deltaT = 8760/time_steps 
# convert the timespan into datetime format
time_index = pd.date_range(start="2022-01-01", periods=time_steps, freq=f"{deltaT}h")

df_merged = pd.DataFrame(index=time_index)
df_merged["Generation"] = df_vGenP["vGenP"].values
df_merged["Curtailment"] = df_vCurtailment["vCurtailment"].values
df_merged["Import_Export"] = df_vImpExp["vImpExp"].values
df_merged["Demand"] = df_merged["Generation"] + df_merged["Import_Export"]


df_plot = df_merged.copy()
df_plot["time"] = df_plot.index
df_plot["stack_top"] = df_plot["Generation"] + df_plot["Import_Export"]
df_plot["PVPotential"] = df_plot["Generation"] + df_plot["Curtailment"]

source = ColumnDataSource(df_plot)

# --- main plot ---
p = figure(
    height=1000,
    width=2500,
    x_axis_type="datetime",
    x_range=(df_plot["time"].min(), df_plot["time"].max()),
    title="Energy Balance",
    tools="xpan,xwheel_zoom,box_zoom,reset,save",
    active_drag="xpan",
    active_scroll="xwheel_zoom"
)

# stacked positive area: Generation
p.varea(
    x="time", y1=0, y2="Generation", source=source,
    fill_color="#fcba03", fill_alpha=0.7, legend_label="PV Generation"
)

p.line(
    x="time", y="PVPotential", source=source,
    line_color="#fcba03", line_width=2, legend_label="PV Potential"
)

# stacked positive area: Import/Export on top of Generation
p.varea(
    x="time", y1="Generation", y2="stack_top", source=source,
    fill_color="#717d86", fill_alpha=0.7, legend_label="Grid"
)

# negative area: Curtailment
p.varea(
    x="time", y1="Generation", y2="PVPotential", source=source,
    fill_color="#fcba032b", fill_alpha=0.7, legend_label="PV Curtailment"
)

# demand as line
p.line(
    x="time", y="Demand", source=source,
    line_color="#000000", line_width=3, legend_label="Demand"
)

# hover tool for inspection
p.add_tools(HoverTool(
    tooltips=[
        ("Time", "@time{%F %H:%M}"),
        ("Generation", "@Generation{0.000} MW"),
        ("Import/Export", "@Import_Export{0.000} MW"),
        ("Curtailment", "@Curtailment{0.000} MW"),
        ("Demand", "@Demand{0.000} MW")
    ],
    formatters={"@time": "datetime"}
))

p.legend.location = "top_left"

# --- range tool (overview bar) ---
select = figure(
    height=200,
    width=2500,
    y_range=p.y_range,
    x_range=(df_plot["time"].min(), df_plot["time"].max()),
    x_axis_type="datetime",
    y_axis_type=None,
    tools="",
    toolbar_location=None
)
select.varea(x="time", y1=0, y2="stack_top", source=source, fill_color="gray", fill_alpha=0.5)

range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2
select.add_tools(range_tool)

# --- widgets for selecting fixed range ---
start_picker = DatePicker(title="Start Date", value=pd.to_datetime("2022-02-14 00:00").date())
duration_input = TextInput(title="Duration (hours)", value="24")
apply_button = Button(label="Apply Range", button_type="primary")

# callback for button click
def update_range():
    try:
        start_date = pd.to_datetime(start_picker.value)
        duration_hours = float(duration_input.value)
        end_date = start_date + pd.Timedelta(hours=duration_hours)
        p.x_range.start = start_date
        p.x_range.end = end_date
    except Exception as e:
        print("Invalid input:", e)

apply_button.on_click(update_range)

# --- layout ---
layout = column(
    row(start_picker, duration_input, apply_button),
    p,
    select
)

curdoc().add_root(layout)
curdoc().title = "RINGS Dashboard"

if __name__ == "__main__":
    #print(df_merged.head())
    print("Starting the Bokeh server...")
