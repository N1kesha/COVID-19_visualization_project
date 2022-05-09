import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import seaborn as sns
import altair as alt
import requests
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d
import plotly.express as px

with st.echo(code_location='below'):
    st.title('COVID-19 worldwide data visualization')

    @st.cache(allow_output_mutation=True)
    def get_data():
        data_url = "covid_data_cleaned.csv"
        return pd.read_csv(data_url)

    df = get_data()
    df1 = pd.read_excel('time_series_covid19_confirmed_global.xls', engine='xlrd')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.dropna(subset=['total_cases', 'total_deaths'])
    df = df.drop(["total_vaccinations", "new_cases", "new_deaths", "total_boosters", "new_vaccinations", "people_vaccinated"], axis=1)

    df['deaths/cases, %'] = df['total_deaths']/df['total_cases']*100
    df['deaths/vaccines'] = df['total_deaths']/df['people_fully_vaccinated']*100
    countries = list(df['location'].unique())
    continents = list(df['continent'].unique())
    continents = [continent for continent in continents if not (pd.isnull(continent)) == True]

    lists = []
    for country in countries:
        x = np.array(df[df['location']==country]['date']).reshape((-1,1))
        y = df[df['location']==country]['deaths/cases, %']
        model = LinearRegression().fit(x, y)
        lists.append([country, round(float(100000000000000000*model.coef_), 2)])

    pares = pd.DataFrame(lists, columns=['country', 'slope'])
    df = df.merge(pares, left_on='location', right_on='country').drop(['country'], axis=1)

    triple = []
    for country in countries:
        triple.append([country, df[df['location']==country]['continent'].unique()[0], df[df['location']==country]['slope'].unique()[0]])
    triple_df = pd.DataFrame(triple, columns=['country', 'continent', 'slope']).dropna()

    """
    The first thing which is interesting to scrutinize is what part of people who got sick with COVID-19 died 
    from it in different countries, and how countries struggled with this mortality problem on all extent from February 2020 to May 2022. 
    The following plot allows you to visually compare what percentage of sick with COVID-19 people died in any countries.
    """

    options_countries = st.multiselect("Choose as many countries (or continents) as you want to plot. You can also type its name in the line.",
                                       countries)
    alt.data_transformers.disable_max_rows()
    df_plot = df[df['location'].isin(options_countries)]

    ### Adapted FROM: (https://altair-viz.github.io/gallery/multiline_tooltip.html)
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['deaths/cases, %'], empty='none')
    line = alt.Chart(df_plot).mark_line(interpolate='basis').encode(
        x='date',
        y='deaths/cases, %',
        color='location')
    selectors = alt.Chart(df_plot).mark_point().encode(
        x='date',
        opacity=alt.value(0),
    ).add_selection(nearest)
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'deaths/cases, %', alt.value(' ')))
    rules = alt.Chart(df_plot).mark_rule(color='gray').encode(
        x='date',
    ).transform_filter(nearest)

    st.altair_chart(alt.layer(
    line, selectors, points, rules, text
    ).properties(width=600, height=500).interactive())

    ### END FROM

    """
    Knowing how graphs for all countries look like, we can build a simple linear regression to 
    compare how effectively and decisively countries have been fighting COVID-19 deaths. Here are 
    the top countries in this indicator:
    
    ### 15 countries which solved the problem of mortality from СOVID-19 the best of all
    """
    st.write(triple_df.sort_values(by=['slope'])[:15].reset_index().drop(['index'], axis=1))

    """
    ### 15 countries which solved the problem of mortality from СOVID-19 the worst of all 
    """
    st.write(triple_df.sort_values(by=['slope'], ascending=False)[:15].reset_index().drop(['index'], axis=1))

    r = requests.get("https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population")
    soup = BeautifulSoup(r.text, features="html.parser")

    population_list = []
    for i in range(2, 243):
        population_list.append([soup.find(class_="wikitable").find_all("tr")[i].find('td').text.strip(),
                                int(soup.find(class_="wikitable").find_all("tr")[i].find_all('td')[
                                        2].text.strip().replace(",", ""))])
    population_df = pd.DataFrame(population_list, columns=['country', 'population'])

    df = df.merge(population_df, left_on='location', right_on='country', how='left').drop(['country'], axis=1)

    df_without_nan = df.copy()
    df_without_nan = df_without_nan.dropna()

    countries_for_vaccines = list(df_without_nan['location'].unique())
    list_perc_of_deaths_and_vaccines = []
    for country in countries_for_vaccines:
        list_perc_of_deaths_and_vaccines.append([country,
                                                 100 * int(
                                                     df[df['location'].isin([country])].tail(1)['total_deaths']) / int(
                                                     df_without_nan[df_without_nan['location'].isin([country])].tail(1)[
                                                         'population']),
                                                 100 * int(
                                                     df_without_nan[df_without_nan['location'].isin([country])].tail(1)[
                                                         'people_fully_vaccinated']) / int(
                                                     df_without_nan[df_without_nan['location'].isin([country])].tail(1)[
                                                         'population']),
                                                 100 * int(
                                                     df_without_nan[df_without_nan['location'].isin([country])].tail(1)[
                                                         'total_cases']) / int(
                                                     df_without_nan[df_without_nan['location'].isin([country])].tail(1)[
                                                         'population'])
                                                 ])

    mortality_and_vaccinated_df = pd.DataFrame(list_perc_of_deaths_and_vaccines,
                                               columns=['country', 'perc. of mortality', 'perc. of vacc', 'perc. of morbidity'])
    mortality_and_vaccinated_df = mortality_and_vaccinated_df.merge(triple_df, left_on='country', right_on='country',
                                                                    how='left')

    """
    The second thing that is curious to know is what part of population in each country 
    from all over the world has died and has vaccinated since February 2020.
    """
    """
    Now you can choose one continent and compare what percentage of population 
    from countries that are on selected continent were vaccinated or died.
    """

    option_continents = st.radio("Choose a continent that you want to consider", continents)
    sns.set(font_scale=0.8)
    sns.set(rc={'figure.figsize': (15,
                                   len(list(mortality_and_vaccinated_df[mortality_and_vaccinated_df['continent'] ==
                                                                        option_continents ]['country'])) / 3)})
    ax = sns.barplot(x="perc. of vacc", y="country",
                data=mortality_and_vaccinated_df[mortality_and_vaccinated_df['continent'] == option_continents].sort_values(
                    by=['perc. of vacc']))
    ax.set_title(option_continents, fontsize=40)
    ax.set_xlabel('Percentage of population that has been vaccinated', fontsize=20);
    ax.set_ylabel('Country', fontsize=20);

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    sns.set(font_scale=0.6)
    sns.set(rc={'figure.figsize': (15,
                                   len(list(mortality_and_vaccinated_df[mortality_and_vaccinated_df['continent'] ==
                                                                        option_continents]['country'])) / 3)})
    ax = sns.barplot(x="perc. of mortality", y="country",
                     data=mortality_and_vaccinated_df[
                         mortality_and_vaccinated_df['continent'] == option_continents].sort_values(
                         by=['perc. of mortality'], ascending=False))
    ax.set_title(option_continents, fontsize=40)
    ax.set_xlabel('Percentage of population that died from COVID-19', fontsize=20);
    ax.set_ylabel('Country', fontsize=20);

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    """
    The third thing that you may be interested in is the morbidity from COVID-19. The following heatmap 
    gives you the opportunity to learn what percentage of population in each country has been infected with Coronavirus.
    The wider a circle is, the larger percentage of the population was infected.
    """
    df1 = df1[['Country', 'Lat', 'Long']]
    mortality_and_vaccinated_df = mortality_and_vaccinated_df.merge(df1, left_on='country', right_on='Country',
                                                                    how='left')
    morbid_list = list(mortality_and_vaccinated_df['perc. of morbidity'])
    circle_radius = interp1d([0, max(morbid_list)], [5, 18])(morbid_list)
    fig = px.density_mapbox(mortality_and_vaccinated_df, lat='Lat', lon='Long', radius=circle_radius, zoom=0,
                            mapbox_style='open-street-map',
                            hover_name='country', hover_data=['perc. of morbidity'],
                            labels={'Long': 'Longitude', 'Lat': 'Latitude',
                                    'perc. of morbidity': '% of morbidity'},
                            title='Percentage of people infected with COVID-19 by 2022', width=700, height=550)
    st.plotly_chart(fig)

    """
    Data sources: 
    1) https://www.kaggle.com/datasets/digvijaysinhgohil/covid19-data-deaths-and-vaccinations?select=covid_data_cleaned.csv
    2) https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
    """














