import pandas as pd
from datetime import datetime
import dill
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings("ignore")


def set_device_os(df):
    # функция заполняющая значения в признаках на основе значений полученных из признака "device_brand"
    device_list = df['device_brand'].unique()
    for i in device_list:
        df_prep = df[df['device_brand'] == f'{i}']
        a = df_prep.device_os.mode()
        if not list(a):
            df_prep.device_os = df_prep.device_os.fillna('other')
            df[df['device_brand'] == f'{i}'] = df_prep
        else:
            a = list(a)
            df_prep.device_os = df_prep.device_os.fillna(f'{a[0]}')
            df[df['device_brand'] == f'{i}'] = df_prep
    return df


def drop_columns(df123):
    # функция с удалением ненужных признаков
    df123123 = df123.drop(columns=['session_id', 'client_id', 'visit_date',
                                   'visit_time', 'visit_number', 'device_model'])
    return df123123


def set_devise(df123):
    # функция для заполнения пропусков в признаказ, в которых не удалось восстановить значения
    df123.device_os = df123.device_os.fillna('other')
    df123.device_brand = df123.device_brand.fillna('other')
    return df123


def fill_data(df123):
    # функция для заполнения пропусков в признаках на основе часто встречающихся значений
    df123.utm_source = df123.utm_source.fillna(df123.utm_source.mode()[0])
    df123.utm_adcontent = df123.utm_adcontent.fillna(df123.utm_adcontent.mode()[0])
    df123.utm_campaign = df123.utm_campaign.fillna(df123.utm_campaign.mode()[0])
    df123.utm_keyword = df123.utm_keyword.fillna(df123.utm_keyword.mode()[0])
    return df123


def utm_concat(df123):
    # объединяем редкие значения в признаках, руководствуясь процентным соотношением редких признаков к
    # общему числу значений
    aa = dict(df123['utm_source'].value_counts()[df123['utm_source'].value_counts() < 2000])
    a1 = list(aa.keys())
    df123.loc[df123['utm_source'].isin(a1), 'utm_source'] = 'rare'
    aa1 = dict(df123['utm_campaign'].value_counts()[df123['utm_campaign'].value_counts() < 10000])
    a2 = list(aa1.keys())
    df123.loc[df123['utm_campaign'].isin(a2), 'utm_campaign'] = 'rare'
    aa2 = dict(df123['utm_medium'].value_counts()[df123['utm_medium'].value_counts() < 10000])
    a3 = list(aa2.keys())
    df123.loc[df123['utm_medium'].isin(a3), 'utm_medium'] = 'rare'
    aa3 = dict(df123['utm_adcontent'].value_counts()[df123['utm_adcontent'].value_counts() < 2000])
    a4 = list(aa3.keys())
    df123.loc[df123['utm_adcontent'].isin(a4), 'utm_adcontent'] = 'rare'
    aa4 = dict(df123['utm_keyword'].value_counts()[df123['utm_keyword'].value_counts() < 2000])
    a5 = list(aa4.keys())
    df123.loc[df123['utm_keyword'].isin(a5), 'utm_keyword'] = 'rare'
    aa4 = dict(df123['device_brand'].value_counts()[df123['device_brand'].value_counts() < 2000])
    a5 = list(aa4.keys())
    df123.loc[df123['device_brand'].isin(a5), 'device_brand'] = 'rare'
    aa4 = dict(
        df123['device_screen_resolution'].value_counts()[df123['device_screen_resolution'].value_counts() < 10000])
    a5 = list(aa4.keys())
    df123.loc[df123['device_screen_resolution'].isin(a5), 'device_screen_resolution'] = 'rare'
    aa4 = dict(df123['device_browser'].value_counts()[df123['device_browser'].value_counts() < 100])
    a5 = list(aa4.keys())
    df123.loc[df123['device_browser'].isin(a5), 'device_browser'] = 'rare'
    aa4 = dict(df123['geo_city'].value_counts()[df123['geo_city'].value_counts() < 70])
    a5 = list(aa4.keys())
    df123.loc[df123['geo_city'].isin(a5), 'geo_city'] = 'rare'
    aa4 = dict(df123['geo_country'].value_counts()[df123['geo_country'].value_counts() < 50])
    a5 = list(aa4.keys())
    df123.loc[df123['geo_country'].isin(a5), 'geo_country'] = 'rare'
    return df123


def main():
    print('hi')
    # основная функция. загружаем  два датафрейма и объединяем их. значения в целевой переменной заменяем на бинарные.
    # Удаляем ненужные признаки и дубликаты. Разделяем выборку на признаки и целевую переменную.
    df = pd.read_csv('data/ga_sessions.csv')
    df1 = pd.read_csv('data/ga_hits.csv')
    df123 = df.merge(df1, how='outer')
    df123 = df123.drop(df123[df123.client_id.isna()].index)
    df123.loc[df123['event_action'].isin(['sub_car_claim_click', 'sub_car_claim_submit_click',
                                          'sub_open_dialog_click', 'sub_custom_question_submit_click',
                                          'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                                          'sub_car_request_submit_click']), 'event_action'] = 1
    df123.loc[df123.event_action != 1, 'event_action'] = 0
    df123 = df123.drop(columns=['hit_date', 'hit_number', 'hit_type',
                                'hit_referer', 'hit_page_path', 'event_category',
                                'event_label', 'event_value', 'hit_time'])
    df123 = df123.drop_duplicates()
    x = df123.drop('event_action', axis=1)
    y = df123['event_action']
    y = y.astype('int')

    # удаляем ненужные датафреймы
    del df
    del df1
    del df123

    categorical_features = make_column_selector(dtype_include=object)

    # объявляем трансофрмеры для признаков и составляем пайплайн
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)])

    preprocessor = Pipeline(steps=[
        ('set_device_os', FunctionTransformer(set_device_os)),
        ('set_devise', FunctionTransformer(set_devise)),
        ('fill_data', FunctionTransformer(fill_data)),
        ('utm_concat', FunctionTransformer(utm_concat)),
        ('drop_columns', FunctionTransformer(drop_columns)),
        ('column_transformer', column_transformer)
    ])

    # объявляем модель случайного леса с заранее найдеными гиперпараметрами
    model = RandomForestClassifier(class_weight='balanced', max_depth=14, max_features='sqrt',
                                   n_estimators=150, min_samples_leaf=3)

    # объявляем составленный пайплайн
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Процесс обучения модели с выводом метрики ROC_AUC и последующим сохранением обученной модели в pkl-файл
    print('modeling...')
    pipe.fit(x, y)
    y_predict = pipe.predict_proba(x)[:, 1]
    metric = roc_auc_score(y, y_predict)
    print(metric)
    with open('cars_pipe_target.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Cars target prediction model',
                'author': 'Gleb Vorobevskiy',
                'version': 1,
                'date': datetime.now(),
                'accuracy': metric
            }
        }, file)


if __name__ == '__main__':
    main()
