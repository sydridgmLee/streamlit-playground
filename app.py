from vega_datasets import data
import streamlit as st
import numpy as np
import pandas as pd

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import altair as alt
from bokeh.plotting import figure
import plotly.figure_factory as ff
import plotly.express as px


################### AREA CHART######################
st.title("Area chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.write(chart_data)
st.area_chart(chart_data)

chart_data = pd.DataFrame(
    {
        "col1": np.random.randn(20),
        "col2": np.random.randn(20),
        "col3": np.random.choice(["A", "B", "C"], 20),
    }
)
st.write(chart_data)
st.area_chart(chart_data, x="col1", y="col2", color="col3")

chart_data = pd.DataFrame(np.random.randn(
    20, 3), columns=["col1", "col2", "col3"])
st.write(chart_data)
st.area_chart(
    # Optional
    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]
)

################### BAR CHART######################
st.title("bar chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.write(chart_data)
st.bar_chart(chart_data)

chart_data = pd.DataFrame(
    {
        "col1": list(range(20)) * 3,
        "col2": np.random.randn(60),
        "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
    }
)
st.write(chart_data)

st.bar_chart(chart_data, x="col1", y="col2", color="col3")

chart_data = pd.DataFrame(
    {"col1": list(range(20)), "col2": np.random.randn(20),
     "col3": np.random.randn(20)}
)
st.write(chart_data)

st.bar_chart(
    # Optional
    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]
)

################### LINE CHART######################
st.title("Line chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.write(chart_data)
st.line_chart(chart_data)

chart_data = pd.DataFrame(
    {
        "col1": np.random.randn(20),
        "col2": np.random.randn(20),
        "col3": np.random.choice(["A", "B", "C"], 20),
    }
)
st.write(chart_data)
st.line_chart(chart_data, x="col1", y="col2", color="col3")

chart_data = pd.DataFrame(np.random.randn(
    20, 3), columns=["col1", "col2", "col3"])
st.write(chart_data)
st.line_chart(
    # Optional
    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]
)

################### scatter chart######################
st.title("scatter chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.write(chart_data)
st.scatter_chart(chart_data)

chart_data = pd.DataFrame(np.random.randn(
    20, 3), columns=["col1", "col2", "col3"])
chart_data['col4'] = np.random.choice(['A', 'B', 'C'], 20)
st.write(chart_data)
st.scatter_chart(
    chart_data,
    x='col1',
    y='col2',
    color='col4',
    size='col3',
)

chart_data = pd.DataFrame(np.random.randn(20, 4), columns=[
                          "col1", "col2", "col3", "col4"])
st.write(chart_data)
st.scatter_chart(
    chart_data,
    x='col1',
    y=['col2', 'col3'],
    size='col4',
    color=['#FF0000', '#0000FF'],  # Optional
)

################### PYPLOT ######################
st.title("Pyplot")
arr = np.random.normal(1, 1, size=100)
st.write(arr)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.pyplot(fig)


################### ALTAIR CHART ######################
st.title("Altair chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.write(chart_data)

c = (
    alt.Chart(chart_data)
    .mark_circle()
    .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
)

st.altair_chart(c, use_container_width=True)


source = data.cars()
st.write(source)
chart = alt.Chart(source).mark_circle().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
).interactive()
st.write(chart)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Altair theme.
    st.altair_chart(chart, theme=None, use_container_width=True)

################### VEGA LITE CHART ######################
st.title("Vega lite chart")
chart_data = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])

st.vega_lite_chart(
    chart_data,
    {
        "mark": {"type": "circle", "tooltip": True},
        "encoding": {
            "x": {"field": "a", "type": "quantitative"},
            "y": {"field": "b", "type": "quantitative"},
            "size": {"field": "c", "type": "quantitative"},
            "color": {"field": "c", "type": "quantitative"},
        },
    },
)

# ################### BOKEH CHART ######################
# st.title("BOKEH CHART")
# x = [1, 2, 3, 4, 5]
# y = [6, 7, 2, 4, 5]

# p = figure(
#     title='simple line example',
#     x_axis_label='x',
#     y_axis_label='y')

# p.line(x, y, legend_label='Trend', line_width=2)

# st.bokeh_chart(p, use_container_width=True)

################### Plotly CHART ######################
st.title("Plotly CHART")
# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
    hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)

st.subheader("Theming")
df = px.data.gapminder()
fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig, theme=None, use_container_width=True)


st.subheader("Define a custom colorscale")
df = px.data.iris()
fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="sepal_length",
    color_continuous_scale="reds",
)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)
