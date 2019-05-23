#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import time
import seaborn as sns


# In[2]:


from bokeh.plotting import figure, output_file, show, ColumnDataSource, gridplot, save
from bokeh.models import HoverTool
#OUTPUT INTO NOTEBOOK
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()


#OUTPUT INTO HTML
#from bokeh.io import output_file, show
#output_file("plot.html")
#save(p)


# In[3]:


#From one notebook: %store data
#Import to another notebook: %store -r data

get_ipython().run_line_magic('store', '-r G')
h = 0.73


# In[4]:


import ipywidgets as widgets
slider = widgets.FloatSlider(min=0, max=10, step=0.5)
slider


# In[5]:


from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider

#output_file("layout_widgets.html")

output_notebook()

# create some widgets
slider = Slider(start=0, end=10, value=1, step=.1, title="Slider")
button_group = RadioButtonGroup(labels=["Option 1", "Option 2", "Option 3"], active=0)
select = Select(title="Option:", value="foo", options=["foo", "bar", "baz", "quux"])
button_1 = Button(label="Button 1")
button_2 = Button(label="Button 2")

show(column(button_1, slider, button_group, select, button_2, width=300))


# In[6]:


#LINKING AND PULLING DATA
get_ipython().run_line_magic('store', '-r Mstellar_central_galaxies_cut')
get_ipython().run_line_magic('store', '-r Mcoldgas_central_galaxies_cut')
get_ipython().run_line_magic('store', '-r Mvir_central_galaxies_cut')

get_ipython().run_line_magic('store', '-r Mstellar_satellite_galaxies_cut')
get_ipython().run_line_magic('store', '-r Mcoldgas_satellite_galaxies_cut')
get_ipython().run_line_magic('store', '-r Mvir_satellite_galaxies_cut')


# In[7]:



x=np.log10( (Mstellar_central_galaxies_cut*10**10))
y0, y1, y2 = np.log10( (Mcoldgas_central_galaxies_cut*10**10)), x, np.log10( (Mcoldgas_central_galaxies_cut*10**10)/(Mstellar_central_galaxies_cut*10**10))

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1, y2=y2))

#Interactive tools
TOOLS = "box_select,lasso_select,help,pan,wheel_zoom"

# create plot and add a renderer
left = figure(tools=TOOLS, width=400, height=400)
left.circle('x', 'y0', color='#1f78b4', size=8,source=source)
left.xaxis.axis_label = 'Mstar'
left.yaxis.axis_label = 'MHI'

# create another plot and add a renderer
right = figure(tools=TOOLS, width=400, height=400)
right.circle('y1', 'y2',color='#1f78b4', size=8,source=source)
right.xaxis.axis_label = 'Mstar'
right.yaxis.axis_label = 'Gas Fraction'

p = gridplot([[left,right]])

show(p)


# In[8]:


select = Select(title="Show on hover:", value="Mvir", options=["Mvir", "Central ID", "HI", "H2"])
show(column(select, width=300))


# In[9]:



source = ColumnDataSource(
        data=dict(
            x=np.log10( (Mstellar_central_galaxies_cut*10**10)),
            y=np.log10( (Mcoldgas_central_galaxies_cut*10**10)),
            desc=G['Mvir'],))

hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(Mstar, MHI )", "($x, $y)"),
            ("Mvir", "@desc"),
        ])

p = figure(plot_width=700, plot_height=700, tools=[hover], title="Centrals")

p.circle('x', 'y', size=8, color='#1f78b4', source=source)
#p.line([8.5,9,10,11], [8.5,9,10,11], line_width=1, line_color='blue')

p.yaxis.axis_label = 'MHI'
p.xaxis.axis_label = 'Mstar'

show(p)


# In[ ]:





# In[10]:


import numpy as np

from bokeh.layouts import grid, column
from bokeh.models import CustomJS, Slider, ColumnDataSource
from bokeh.plotting import figure, output_file, show

#output_file('dashboard.html')

tools = 'pan'

#hover tool
hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(Mstar, MHI )", "($x, $y)"),
            ("Mvir", "@desc"),
        ])
#input data
source = ColumnDataSource(
        data=dict(
            x = np.log10( (Mstellar_central_galaxies_cut*10**10)),
            y = np.log10( (Mcoldgas_central_galaxies_cut*10**10)),
            desc = G['Mvir'],))


#amp_slider = Slider(start=min(np.log10(G['Mvir']*1e10/h)), end=max(np.log10(G['Mvir']*1e10/h)), 
#                        value=min(np.log10(G['Mvir']*1e10/h)), step=0.5, title=r"Mvir", 
#                        callback=callback, callback_policy='mouseup')

    
plot = figure(plot_width=400, plot_height=400, tools=[hover])
plot.circle('x', 'y', source=source, color='#1f78b4')

slider = Slider(start=min(np.log10(G['Mvir']*1e10/h)), end=max(np.log10(G['Mvir']*1e10/h)), 
                        value=min(np.log10(G['Mvir']*1e10/h)), step=0.5, title=r"Mvir")

callback = CustomJS(args=dict(source=source, slider=slider), code="""
    var data = source.data;
    var f = slider.value;
    x = data['x']
    y = data['y']
    
    for (i = 0; i < x.length; i++) {
        y[i] = (x[i] + f)
    }
    
    
    // necessary becasue we mutated source.data in-place
    source.change.emit();
""")
slider.js_on_change('value', callback)


show(column(slider, plot))


# In[ ]:





# In[ ]:




