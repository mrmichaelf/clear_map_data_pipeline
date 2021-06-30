from joblib import Parallel, delayed
from itertools import zip_longest
import multiprocessing
import geopandas as gp
import pandas as pd
import numpy as np
import requests
import datetime
import boto3
import json
import time
import csv
import os
import io


def apply_parallel(df_grouped, func):
    ret_lst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(ret_lst)


def rebase_15(df):
    for index, row in df.iterrows():  # this is unique for the moh data format, and this loop is taking time
        if df.loc[df['date'] <= row['date']]['cases'].sum() == 0:  # if up to this date, for this area all 0, so calculate
            df.at[index, 'm_cases'] = df.loc[df['date'] <= row['date']]['new_case'].sum()  # sum of days with new case
        if df.loc[df['date'] <= row['date']]['vaccine'].sum() == 0:  # if up to this date, for this area all 0, so calculate
            df.at[index, 'm_vaccine'] = df.loc[df['date'] <= row['date']]['new_vaccine'].sum()  # sum of days with new vaccine
        # if df.loc[df['date'] <= row['date']]['tests'].sum() == 0:  # 23.12 change set made it irrelevant. if up to this date, for this area all 0, so calculate
        #     df.at[index, 'm_tests'] = df.loc[df['date'] <= row['date']]['new_test'].sum()  # 23.12 change set made it irrelevant. sum of days with new test
    return df


def parse():
    # region input
    in_shape_file_path = '/Users/michael/Downloads/Data/areas.geojson'
    out_file_path = '/Users/michael/Downloads/Data/'
    url = 'https://data.gov.il/datastore/dump/d07c0771-01a8-43b2-96cc-c6154e7fa9bd?bom=true'
    headers = {
        'path': '...',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
        'Cookie': '...'
    }
    # response = requests.get(url=url, headers=headers)  # TODO - comment out is temporary due to changes on MOH server - they have some bug: following 3 lines should be the long term
    # decoded_content = response.content.decode('utf-8')
    # data = pd.read_csv(io.StringIO(decoded_content))
    data = pd.read_csv('/Users/michael/Downloads/geographic-sum-per-day-ver_00310.csv')  # TODO - this is temporary
    ###
    # import urllib.request  # TODO - this was a try to overcome the MOH server bug, not worked
    # url = 'https://data.gov.il/api/3/action/datastore_search?resource_id=d07c0771-01a8-43b2-96cc-c6154e7fa9bd'
    # response = urllib.request.urlopen(url).read().decode('UTF-8')
    ###
    data = data.drop(['accumulated_tested', 'new_tested_on_date', '_id', 'accumulated_recoveries', 'new_recoveries_on_date', 'accumulated_hospitalized', 'new_hospitalized_on_date', 'accumulated_deaths', 'new_deaths_on_date', 'town', 'new_diagnostic_tests_on_date', 'accumulated_vaccination_second_dose', 'new_vacc_second_dose_on_date'], axis=1, errors='ignore')  # remove unused columns
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')  # so we have valid date object
    data.columns = ['city', 'area', 'date', 'cases', 'new_case', 'tests', 'vaccine', 'new_vaccine']  # rename columns. 23.12 change set made 'new_test' irrelevant
    data.loc[data['area'].isna(), 'area'] = 0  # if there is no area code - make 0 as default
    data = data.astype({'area': int})  # convert area code to int
    data.loc[data['city'].isna(), 'city'] = 0  # if there is no city code - make 0 as default
    data.drop(data.loc[(data['city'] == 0) & (data['area'] == 0)].index, inplace=True)  # remove nan cities
    data['id'] = data['city'] * 10000 + data['area']  # add unique area id
    data.loc[data['cases'] == '<15', 'cases'] = 0  # convert text to numbers
    data.loc[data['tests'] == '<15', 'tests'] = 0  # convert text to numbers
    data.loc[data['vaccine'] == '<15', 'vaccine'] = 0  # convert text to numbers
    data = data.astype({'cases': int})  # as ints...
    data = data.astype({'tests': float})  # as ints...
    data = data.astype({'vaccine': float})  # as ints...
    data.loc[data['new_case'] == 'TRUE', 'new_case'] = 1  # convert text to numbers
    data.loc[data['new_case'] == 'FALSE', 'new_case'] = 0  # convert text to numbers
    data.loc[data['new_vaccine'] == 'TRUE', 'new_vaccine'] = 1  # convert text to numbers
    data.loc[data['new_vaccine'] == 'FALSE', 'new_vaccine'] = 0  # convert text to numbers
    # data.loc[data['new_test'] == 'TRUE', 'new_test'] = 1  # 23.12 change set made it irrelevant convert text to numbers
    # data.loc[data['new_test'] == 'FALSE', 'new_test'] = 0  # 23.12 change set made it irrelevant convert text to numbers
    data = data.astype({'new_case': int})  # as ints...
    data = data.astype({'new_vaccine': int})  # as ints...
    # data = data.astype({'new_test': int})  # 23.12 change set made it irrelevant as ints...
    data = data.sort_values('date', ascending=False)  # first sorting then grouping ensures each group internally will be sorted
    shape = gp.read_file(in_shape_file_path)  # read the areas file (static)
    dates_df = []  # for dates csv for s3 and then for JS
    last_date = data['date'].iloc[0].to_pydatetime()
    weeks_1 = last_date - datetime.timedelta(days=7 * 1)
    weeks_2 = last_date - datetime.timedelta(days=7 * 2)
    weeks_3 = last_date - datetime.timedelta(days=7 * 3)
    names = [('all', 0, 0), ('wave_2', weeks_3.month, weeks_3.day), ('weeks_2', weeks_2.month, weeks_2.day), ('weeks_1', weeks_1.month, weeks_1.day)]
    # endregion
    # region make order in areas and cities
    # case 1: city without area 0 in cases - do nothing
    # case 2: city with area 0 and other areas in cases - remove area 0
    # case 3: city with only area 0 in cases AND more areas in general shape - union all shapes under city name (no area name)
    # case 4: city with only area 0 in cases AND only single area in general shape - do nothing, set area id to 1
    for city, group in data.groupby(['city']):
        # case 1
        if len(group.loc[group['area'] == 0]) > 0:
            # case 2
            if len(group.loc[group['area'] != 0]) != 0:
                data.drop(group.loc[group['area'] == 0].index, inplace=True)  # remove area 0
            # case 3
            elif shape.loc[shape['city'] == city]['area'].nunique() > 1:
                dissolved = shape.loc[shape['city'] == city].unary_union
                if dissolved.geom_type == 'MultiPolygon':
                    shape.loc[shape['city'] == city, 'geometry'] = shape.loc[shape['city'] == city].unary_union.convex_hull
                else:
                    shape.loc[shape['city'] == city, 'geometry'] = shape.loc[shape['city'] == city].unary_union  # the first area with id=1 (same as 0 got in cases data)
                data.loc[data['city'] == city, 'area'] = group['area'].iloc[0] + 1  # to be the same like in shape
                data.loc[data['city'] == city, 'id'] = group['id'].iloc[0] + 1
                shape.loc[shape['city'] == city, 'areas_name'] = shape.loc[shape['city'] == city]['name'].iloc[0]  # change area name to be city name, keep area & id (will represent entire city)
                shape.loc[shape['city'] == city, 'pop'] = shape.loc[shape['city'] == city]['pop'].sum()
                shape.loc[shape['city'] == city, 'area'] = group['area'].iloc[0] + 1  # to be the same like in shape
                shape.loc[shape['city'] == city, 'id'] = group['id'].iloc[0] + 1
                if shape.loc[shape['city'] == city, 'rank'].notnull().values.any():
                    shape.loc[shape['city'] == city, 'rank'] = shape.loc[shape['city'] == city]['rank'].median()
                shape.loc[shape['city'] == city] = shape.loc[shape['city'] == city].drop_duplicates()
            # case 4
            else:
                data.loc[data['city'] == city, 'area'] = group['area'].iloc[0] + 1  # to be the same like in shape
                data.loc[data['city'] == city, 'id'] = group['id'].iloc[0] + 1  # since city has only area 0 - all id's are the same
    # endregion
    # region process
    shape.drop(shape.loc[shape['id'].isna()].index, inplace=True)  # remove areas that exists in shape but not exists in MOH data
    shape = shape.astype({'id': int})  # later will be merged on this field so should be consist with data - as int
    data['m_cases'] = 0  # update areas with 0 cases to be number of new cases, means: if cases less than 15, so at least this
    data['m_vaccine'] = 0  # update areas with 0 vaccine to be number of new vaccines, means: if vaccines less than 15, so at least this
    # data['m_tests'] = 0  # 23.12 change set made it irrelevant. update areas with 0 tests to be number of new tests, means: if tests less than 15, so at least this
    data = apply_parallel(data.groupby('id'), rebase_15)
    data['cases'] += data['m_cases']  # will influence only those with 0 cases
    data['cases'] += data['m_vaccine']  # will influence only those with 0 vaccines
    # data['tests'] += data['m_tests']  # 23.12 change set made it irrelevant. will influence only those with 0 tests
    data = data.merge(shape.set_index('id'), left_on='id', right_on='id')  # merge with shape
    data = data.drop(['city_x', 'area_x', 'new_case', 'm_cases', 'new_vaccine', 'm_vaccine', 'city_y', 'area_y', 'name'], axis=1, errors='ignore')  # unused columns. 23.12 change set made 'new_test', 'm_tests' irrelevant.
    data = data.sort_values('date', ascending=True)  # sort by ascending order - important for later
    # endregion
    # region ramzor & percent
    # data['ramzor'] = 0
    data['percent'] = 0
    data['p_vaccine'] = 0
    for id_, group in data.groupby(by='id'):
        data.loc[data['id'] == id_, 'percent'] = np.round((data.loc[data['id'] == id_].iloc[-1]['cases'] / data.loc[data['id'] == id_].iloc[-1]['pop']) * 100)  # constant percent for the last date - biggest...
        data.loc[data['id'] == id_, 'p_vaccine'] = np.round((data.loc[data['id'] == id_].iloc[-1]['vaccine'] / data.loc[data['id'] == id_].iloc[-1]['pop']) * 100)  # constant percent for the last date - biggest...
    data.loc[data['p_vaccine'] > 100, 'p_vaccine'] = 100
    #     weekly_normalized_cases = (data.loc[data['id'] == id_]['cases'].rolling(7).sum()/data.loc[data['id'] == id_]['pop'])*10000  # N
    #     prev_week_normalized_cases = weekly_normalized_cases.shift(periods=7, fill_value=0)
    #     growth_rate = weekly_normalized_cases/prev_week_normalized_cases  # G
    #     positive_test_rate = (data.loc[data['id'] == id_]['cases'].rolling(7).sum())/(data.loc[data['id'] == id_]['tests'].rolling(7).sum())  # P
    #     data.loc[data['id'] == id_, 'ramzor'] = 2 + np.log(weekly_normalized_cases * growth_rate * growth_rate) + positive_test_rate/8
    # data['ramzor'] = data['ramzor'].fillna(0)
    # data.loc[data['ramzor'] < 0, 'ramzor'] = 0
    # data.loc[data['ramzor'] > 10, 'ramzor'] = 10
    # data['ramzor'] = data['ramzor'].round(1)
    data = data.sort_values('date', ascending=False)  # sort by descending order - important for later
    # endregion
    # region create geojson files
    for name in names:  # for each time filter
        data_x = pd.DataFrame()  # temporary data frame to copy the original data since used on for loop
        if name[0] == 'all':  # all historical data - align to weekly basis - will be executed first and once
            for name_, group in data.groupby(by='id'):
                data_x = data_x.append(group.iloc[::7, :].copy(), ignore_index=True)  # because sort is date dsc - we keep the end date fixed
        else:  # all other periods - will be executed several times in a raw after all
            data.drop(data.loc[data['date'] < datetime.datetime(year=2021, month=name[1], day=name[2])].index, inplace=True)  # TODO hard coded year
            data_x = data.copy()
        for uid, group in data_x.groupby(['id']):  # rebase for zero cases and tests at the beginning of the period
            data_x.loc[data_x['id'] == uid, 'cases'] = group['cases'] - group.iloc[-1]['cases']
            data_x.loc[data_x['id'] == uid, 'tests'] = group['tests'] - group.iloc[-1]['tests']
        # data_x.drop(data_x.loc[data_x['cases'] == 0].index, inplace=True)  # No need, so 0 will be displayed as transparent (!!!) by definition because of rebasing last date will be 0 and dropped. More accurately: data_x.loc[data_x['id'] == uid] = data_x.loc[data_x['id'] == uid][:-1]
        data_x['delta'] = 0  # create daily deltas for daily mode
        for uid, group in data_x.groupby(['id']):
            data_x.loc[data_x['id'] == uid, 'delta'] = group['cases'].diff(periods=-1).fillna(group['cases'])
        data_x['normalized'] = np.round((data_x['cases'] / data_x['pop']) * 100000)  # now when the numbers are correct - calculate normalized
        data_x = data_x.astype({'normalized': int})  # convert to int
        dates_df.append(pd.Series(data_x['date'].dt.date.unique()).to_list())  # for dates csv - list unique, by definition they already sorted
        dates_df.append([int(y) for y in data_x['normalized'].quantile(np.arange(0.05, 1, 0.05).tolist()).tolist()])  # create quantiles for colors
        dates_df.append(data_x.groupby('date', as_index=False)['cases'].sum()['cases'].diff().fillna(data_x.groupby('date', as_index=False)['cases'].sum()['cases']))  # daily sums
        shape = gp.GeoDataFrame(data_x, geometry=data_x['geometry']).copy()  # create the gp data frame object
        shape = shape.drop(['pop', 'vaccine'], axis=1, errors='ignore')  # unused columns (before daily display, ID was also dropped)
        shape.columns = ['date', 'num_cases', 'tests', 'id', 'socio_economic_rank', 'area_name', 'geometry', 'percent', 'p_vaccine', 'delta', 'normalized']
        shape.to_file(out_file_path + name[0] + '.geojson', driver='GeoJSON', encoding='UTF-8')  # export the polygons geojson file
        shape['geometry'] = shape['geometry'].boundary  # convert to lines
        shape[['date', 'num_cases', 'geometry']].to_file(out_file_path + name[0] + '_lines.geojson', driver='GeoJSON', encoding='UTF-8')  # export the lines geojson file
    # endregion
    # region create mbtiles using tippecanoe
    os.chdir(out_file_path)
    for name in names:
        os.system('tippecanoe -zg -f -o il_' + name[0] + '.mbtiles -L areas:' + name[0] + '.geojson -L areas_lines:' + name[0] + '_lines.geojson')
    # endregion
    # region upload tile set to Mapbox
    api_key = '...'
    stage_str = 'https://api.mapbox.com/uploads/v1/mikethe1/credentials?access_token=' + api_key
    upload_str = 'https://api.mapbox.com/uploads/v1/mikethe1?access_token=' + api_key
    for name in names:
        response = requests.post(stage_str).json()
        s3_client = boto3.client('s3', aws_access_key_id=response['accessKeyId'], aws_secret_access_key=response['secretAccessKey'], aws_session_token=response['sessionToken'], region_name='us-east-1')
        s3_client.upload_file(out_file_path + 'il_' + name[0] + '.mbtiles', response['bucket'], response['key'])
        headers = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'
        }
        data = {
            'url': 'http://' + response['bucket'] + '.s3.amazonaws.com/' + response['key'],
            'tileset': 'mikethe1.il3_' + name[0] + '_' + str(dates_df[6][0].day)
        }
        m_response = requests.post(url=upload_str, headers=headers, data=json.dumps(data)).json()
        print(m_response)
    # endregion
    # region upload dates csv to s3
    with open(out_file_path + 'dates_array.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['all', 'month', 'weeks_2', 'weeks_1', 'colors_all', 'colors_month', 'colors_weeks_2', 'colors_weeks_1', 'sum_all', 'sum_month', 'sum_weeks_2', 'sum_weeks_1'])
        dates_df[0].reverse()
        dates_df[3].reverse()
        dates_df[6].reverse()
        dates_df[9].reverse()
        dates_df_x = [dates_df[0], dates_df[3], dates_df[6], dates_df[9],
                      dates_df[1], dates_df[4], dates_df[7], dates_df[10],
                      dates_df[2], dates_df[5], dates_df[8], dates_df[11]]
        for values in zip_longest(*dates_df_x):
            writer.writerow(values)
    s3_client = boto3.client('s3', aws_access_key_id='...', aws_secret_access_key='...', region_name='eu-west-2')
    s3_client.upload_file(out_file_path + 'dates_array.csv', 'clearmap', 'il_dates.csv', ExtraArgs={'CacheControl': 'max-age=60'})
    # endregion


if __name__ == "__main__":
    start_time = time.time()
    parse()
    print('Time: ' + str(datetime.timedelta(seconds=(time.time() - start_time))))
