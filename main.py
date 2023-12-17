import pandas as pd
import numpy as np

import streamlit as st
import seaborn as sns
import altair as alt
import requests
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('COVID-19 worldwide data visualization')

df = pd.read_csv("covid_data_cleaned.csv")
df1 = pd.read_excel('time_series_covid19_confirmed_global.xls', engine='xlrd')

df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.dropna(subset=['total_cases', 'total_deaths', 'people_fully_vaccinated'], inplace=True)

df['deaths_per_case'] = df['total_deaths'] / df['total_cases']
df['deaths_per_vaccine'] = df['total_deaths'] / df['people_fully_vaccinated']
countries = list(df['location'].unique())
continents = list(df['continent'].unique())
continents = [continent for continent in continents if not (pd.isnull(continent)) == True]


"""
The first thing which is interesting to scrutinize is what part of people who got sick with COVID-19 died 
from it in different countries, and how countries struggled with this mortality problem on all extent from February 2020 to May 2022. 
The following plot allows you to visually compare what percentage of sick with COVID-19 people died in any countries.
"""

options_countries = st.multiselect(
    "Choose as many countries (or continents) as you want to plot. You can also type its name in the line.",
    countries, default = ['Russia', 'Spain', 'Germany'])
alt.data_transformers.disable_max_rows()
df_plot = df[df['location'].isin(options_countries)]

### FIRST PLOT ###
nearest = alt.selection_point(nearest=True, on='mouseover', fields=['deaths_per_case'], empty=False)
line = alt.Chart(df_plot).mark_line(interpolate='basis').encode(
    x='date',
    y='deaths_per_case',
    color='location')
selectors = alt.Chart(df_plot).mark_point().encode(
    x='date',
    opacity=alt.value(0)).add_params(nearest)
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'deaths_per_case', alt.value(' ')))
rules = alt.Chart(df_plot).mark_rule(color='gray').encode(
    x='date',
).transform_filter(nearest)

st.altair_chart(alt.layer(
    line, selectors, points, rules, text
).properties(width=595, height=500).interactive())
### END OF FIRST PLOT ###



r = requests.get("https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population")
soup = BeautifulSoup(r.text, features="html.parser")

population_list = []
for i in range(2, 243):
    population_list.append([soup.find(class_="wikitable").find_all("tr")[i].find('td').text.strip(),
                            int(soup.find(class_="wikitable").find_all("tr")[i].find_all('td')[1]. \
                                text.strip().replace(",", ""))])

population_df = pd.DataFrame(population_list, columns=['location', 'population'])
df = df.merge(population_df, on='location', how='inner')

countries_for_vaccines = df.location.unique().tolist()
deaths_vaccines_cases_perc = []
for country in countries_for_vaccines:
    deaths_vaccines_cases_perc.append([country,
                                       float(100 * df[df['location'] == country]['total_deaths'].iloc[-1] /
                                             df[df['location'] == country]['population'].iloc[-1]),
                                       float(100 * df[df['location'] == country]['people_fully_vaccinated'].iloc[-1] /
                                           df[df['location'] == country]['population'].iloc[-1]),
                                       float(100 * df[df['location'] == country]['total_cases'].iloc[-1] /
                                             df[df['location'] == country]['population'].iloc[-1])
                                       ])

mort_vacc_morb_df = pd.DataFrame(deaths_vaccines_cases_perc,
                                 columns=['location', 'mortality_rate', 'vaccinated_rate', 'morbidity_rate'])

mort_vacc_morb_df = mort_vacc_morb_df.merge(
    df[['location', 'continent']].groupby(by='location', as_index=False).min(),
    on='location')


st.write(""" \n\n
The second thing that is curious to know is what part of population in each country 
from all over the world has died and has vaccinated since February 2020.

Now you can choose one continent and compare what percentage of population 
from countries that are on selected continent were vaccinated or died.
""")

### SECOND PLOT ###
option_continents = st.radio("Choose a continent that you want to consider", continents)

x = list(mort_vacc_morb_df[mort_vacc_morb_df['continent'] == option_continents]['location'])
sns.set(rc={'figure.figsize': (20, len(x) / 2)})
palette = sns.color_palette("Reds_r", len(x))

ax = sns.barplot(x="mortality_rate", y="location", palette = palette,
                 data=mort_vacc_morb_df[
                     mort_vacc_morb_df['continent'] == option_continents].sort_values(
                     by=['mortality_rate'], ascending=False))
ax.set_title(option_continents, fontsize=60)
ax.set_xlabel('Percentage of population that died from COVID-19', fontsize=30)
ax.set_ylabel('Country', fontsize=40)
plt.yticks(fontsize=20)
plt.xticks(fontsize=30)

descript_stats = mort_vacc_morb_df[mort_vacc_morb_df['continent'] == option_continents]['mortality_rate'].describe()
st.subheader('Percentage of population that died from COVID-19')
st.write(pd.DataFrame(descript_stats).T)
st.pyplot()


axs = sns.barplot(x="vaccinated_rate", y="location", palette=palette,
                 data=mort_vacc_morb_df[
                     mort_vacc_morb_df['continent'] == option_continents].sort_values(by=['vaccinated_rate'], ascending=False))

axs.set_title(option_continents, fontsize=60)
axs.set_xlabel('Percentage of population that has been vaccinated', fontsize=30)
axs.set_ylabel('Country', fontsize=40)
plt.yticks(fontsize=20)
plt.xticks(fontsize=30)

descript_stats = mort_vacc_morb_df[mort_vacc_morb_df['continent'] == option_continents]['vaccinated_rate'].describe()


st.subheader('Percentage of population that has been vaccinated')
st.write(pd.DataFrame(descript_stats).T)
st.pyplot()
### END OF SECOND PLOT ###


st.subheader('Percentage of people infected with COVID-19 by 2022')

"""
The third thing that you may be interested in is the morbidity from COVID-19. The following heatmap 
gives you the opportunity to learn what percentage of population in each country has been infected with Coronavirus.
The wider a circle is, the larger percentage of the population was infected.
"""

df1 = df1[['Country', 'Lat', 'Long']]
mort_vacc_morb_df = mort_vacc_morb_df.merge(df1, left_on='location', right_on='Country', how='left')
mort_vacc_morb_df['morbidity_rate'] = round(mort_vacc_morb_df['morbidity_rate'], 1)

### THIRD PLOT ###
morbid_list = list(mort_vacc_morb_df['morbidity_rate'])
circle_radius = interp1d([0, max(morbid_list)], [5, 18])(morbid_list)
fig = px.density_mapbox(mort_vacc_morb_df, lat='Lat', lon='Long', radius=circle_radius, zoom=0,
                        color_continuous_scale= [
                                        [0.0, "green"],
                                        [0.5, "green"],
                                        [0.51111111, "yellow"],
                                        [0.71111111, "yellow"],
                                        [0.71111112, "red"],
                                        [1, "red"]],
                        mapbox_style='open-street-map',
                        hover_name='location', hover_data=['morbidity_rate'],
                        labels={'Long': 'Longitude', 'Lat': 'Latitude',
                                'morbidity_rate': '% of morbidity'},
                        width=850, height=610)

st.plotly_chart(fig)
### END OF THIRD PLOT ###