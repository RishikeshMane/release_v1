import streamlit as st 
import numpy as np         
import pandas as pd 
import plotly.express as px 
import altair as alt
from transformers import pipeline , AutoModelForQuestionAnswering , AutoTokenizer
import matplotlib
import matplotlib as plt
from matplotlib.backends.backend_agg import RendererAgg
import plotly.figure_factory as ff
# import pygwalker as pyg
import streamlit.components.v1 as components
from transformers import pipeline
import base64
from PIL import Image
import os
import plotly.graph_objects as go
#os.environ["OPENAI_API_KEY"] = ""

OpenAI_key = os.environ.get("OPEN_AI_KEY")
#lida content code 
from lida.datamodel import Goal
from lida import llm
from lida import Manager, TextGenerationConfig
text_gen = llm("openai") # for openai
lida = Manager(text_gen = llm("openai", api_key=OpenAI_key)) # !! api key
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)



df = pd.read_csv("file5.csv")
#streamlit
st.set_page_config(
    page_title = 'Azure ETL with Streamlit',
    page_icon = '✅',
    layout = 'wide'
)
st.title("Data Analytics and Summarization with LLM's ")

job_filter = st.selectbox("Select the Company name", pd.unique(df['name']))
#job_filter1 = st.selectbox("Select the Company name", pd.unique(df['name']),key="one")
lenofdf=len(df)
listofnames=['bank','hospital']
name_filter = st.selectbox("Select types from dropdown to see metrics below ",listofnames)
#col1, col2, col3 = st.columns(3)
#col1.metric("Total Df count",lenofdf)
#col2.metric("t2", "..", "process..")
#col3.metric("t3", "..", "process..")
oversubdata=df[df['name'].str.lower().str.contains(name_filter)]
#host_data=df[df['name'].str.lower().str.contains(name_filter)]
lenofover_data = len(oversubdata)
col1, col2, col3 = st.columns(3)
cal_for_name_filter = lenofdf - lenofover_data
with col1:
    st.metric("Total Df count",lenofdf)
with col2:
   st.metric('Total count of '+ name_filter,lenofover_data,-cal_for_name_filter)
ar = np.arange(1, 100)
rand_filter = st.selectbox("select the Numbers", ar)
df_sin=df[:rand_filter]

# model_name = "deepset/roberta-large-squad2"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# nlp = pipeline('question-answering', model=model_name, tokenizer=tokenizer)
# #st.snow()
# with st.container():
#     df = df[df['name']==job_filter]
#     df_save = df['description']
#     #st.dataframe(df_save)
#     text_inp = str(df_save)
#     prompt = st.chat_input("Enter you aspect question")
#     if prompt:
#         QA_input = {
#                 'question': prompt,
#                 'context': text_inp
#                 }
#         res = nlp(QA_input)
#         st.write(f"User has sent the following prompt: {res}")
                
                













#listofnames=['bank','hospital']
#name_filter = st.selectbox("Types",listofnames)

bank_data=df_sin[df_sin['name'].str.lower().str.contains(name_filter)]



# df['text'] = df['text'].astype(str).str.lower()
# df.head(3)

# creating a single-element container.
placeholder = st.empty()

df = df[df['name']==job_filter]
df_save = df['ceo_approval']

data = {"name":df_sin["name"],"approval":df_sin["ceo_approval"]}



source = pd.DataFrame({
        'rating': df_sin['rating'],
        'company_name': df_sin['name']
     })
 
line_chart = alt.Chart(source).mark_line().encode(
        y='rating',
        x='company_name',
    )


source1 = pd.DataFrame({
        'ceo_approval1': df_sin['ceo_approval'],
        'company_name1': df_sin['name']
     })
 
line_chart1 = alt.Chart(source1).mark_line().encode(
        y='ceo_approval1',
        x='company_name1',
    )



# matplotlib.use("agg")
# _lock = RendererAgg.lock


rt_data   = df_sin['rating']
name_data =df_sin['name']

#merge the two dataframe to get a column with the color
df_sink = pd.merge(rt_data, name_data,how = 'cross')
# colors = df['color'].tolist()
print(df_sink[:5])
# row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns((0.2, 1, .2, 1, .2))
# with row0_1, _lock:
#     st.header("Political parties")
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(rt_data, labels=(rt_data.index + ' (' + rt_data.map(str)
#     + ')'), wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white'
#     }, colors=colors)
#     #display a white circle in the middle of the pie chart
#     p = plt.gcf()
#     p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
#     st.pyplot(fig)


# fig = px.pie(df_sin, values='rating', names='name', title='Population of European continent')
# fig.update_traces(hole=.4, hoverinfo="label+percent+name")

# fig = px.scatter(
#         df_sin,
#         x="rating",
#         y="name",
#         size="name",
#         color="continent",
#         log_x=True,
#         size_max=60,
#         template="plotly_dark",
#         title="Gapminder 2007: 'plotly_dark' theme",
#     )
# group_labels = ['Group 1', 'Group 2']
# colors = ['#333F44','#94F3E4']

#     # Create distplot with curve_type set to 'normal'

# fig = ff.create_distplot(df_sink, group_labels, show_hist=False, colors=colors)

#     # Add title
# fig.update_layout(title_text='Curve and Rug Plot')
# st.plotly_chart(fig, theme="streamlit")
 
# with placeholder.container():
#         st.title("Plotly graph")

#         fig = go.Figure(data=go.Scatter(x=name_data, y=rt_data))
#         fig.update_layout(
#             title="name vs approval",
#             xaxis_title="name of company",
#             yaxis_title="rating data",
#             template="plotly_white"
#         )
#         st.plotly_chart(fig)

fig = go.Figure(data=go.Scatter(x=name_data, y=rt_data))
fig.update_layout(
            title="name vs approval",
            xaxis_title="name of company",
            yaxis_title="rating data",
            template="plotly_white"
        )

 
 
 
#comment out below code to display chart agazin 

expander_forexcel1 = st.expander("See Graph 1")
expander_forexcel1.write(fig)

expander_forexcel2 = st.expander("See Graph 2")
expander_forexcel2.write(line_chart1)

# st.altair_chart(line_chart, use_container_width=True)
# st.altair_chart(line_chart1, use_container_width=True)

csv_data = df_sin.to_csv()

# df_sin.to_csv('data.csv', index=False)



data_final = "file5.csv"

df_pass = data_final[:10]




summary = lida.summarize(df_pass,summary_method="default",
        textgen_config=textgen_config)

import io

# goals can also be based on a persona 
persona = "person who wants to get best ceo_approval and rating"
personal_goals = lida.goals(summary, n=10, persona=persona, textgen_config=textgen_config)
for goal in personal_goals:
    st.write(goal)

ar1 = np.arange(0,9)
rand_filter1 = st.selectbox("select the Number for graph", ar1)

i = rand_filter1
library = "seaborn"
textgen_config = TextGenerationConfig(n=20, temperature=0.2, use_cache=True)
charts = lida.visualize(summary=summary, goal=personal_goals[i], textgen_config=textgen_config, library=library)  
img_base64_string = charts[0].raster
imgdata = base64.b64decode(img_base64_string)
img = Image.open(io.BytesIO(imgdata))
st.image(img)


# goals = lida.goals(summary, n=10, textgen_config=textgen_config)
# for goal in goals:
#     st.write(goal)


#st.text(bank_data[:10])
# i = 3
# library = "plotly"
# textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
# charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
# img_base64_string = charts[0].raster
# import io


# imgdata = base64.b64decode(img_base64_string)
# img = Image.open(io.BytesIO(imgdata))
# st.image(img)



st.subheader("Query your Data to Generate Graph")
    
text_area = st.text_area("Query your Data to Generate Graph", height=200)
if st.button("Generate Graph"):
    if len(text_area) > 0:
        st.info("Your Query: " + text_area)
        lida = Manager(text_gen = llm("openai")) 
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        summary = lida.summarize(df_pass, summary_method="default", textgen_config=textgen_config)
        user_query = text_area
        charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
        charts[0]
        image_base64 = charts[0].raster
        imgdata = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(imgdata))
        st.image(img)

expander_forexcel = st.expander("See Excel Data")
expander_forexcel.write(df_sin)
# pyg_html = pyg.to_html(df_sin)

# st.dataframe(df_sin)

