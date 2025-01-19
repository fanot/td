import pandas as pd
import matplotlib.pyplot as plt

# Чтение файла
well = 'FN-SS-KP-12-103'
data_path = f'train/{well}_merged_well_data.csv'
df = pd.read_csv(data_path)

# Преобразуем даты в datetime формат
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')

# Определяем временные интервалы для каждой скважины исторической группы
well_intervals = {
    'P12': [('2013-01-01', '2014-06-30'), ('2017-01-01', '2018-06-30')],
    'P14': [('2013-07-01', '2014-12-31'), ('2017-07-01', '2018-12-31')],
    'P16': [('2014-01-01', '2015-04-30'), ('2018-01-01', '2020-06-30')],
    'P18': [('2014-07-01', '2015-12-31'), ('2018-07-01', '2019-12-31')],
    'P20': [('2013-01-01', '2015-01-31'), ('2017-01-01', '2019-01-31')],
    'P22': [('2013-06-01', '2015-06-30'), ('2017-06-01', '2019-06-30')],
    'P24': [('2013-12-01', '2015-12-31'), ('2017-12-01', '2019-12-31')]
}

# Интервал для прогнозной группы
forecast_start = pd.to_datetime('2018-01-01')
forecast_end = pd.to_datetime('2020-12-31')

# Списки скважин
wells_to_zero_history = ['P12', 'P14', 'P16', 'P18', 'P20']
wells_to_zero_forecast = ['P11', 'P13','P15', 'P17', 'P19']

pressure_column = 'Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'
debit_column = 'Дебит нефти (И), ст.бр/сут'

# Обнуляем значения для каждой скважины из исторической группы
for well in wells_to_zero_history:
    for period_start, period_end in well_intervals[well]:
        mask = (
            (df['Well Name'] == well) & 
            (df['Дата'] >= pd.to_datetime(period_start)) & 
            (df['Дата'] <= pd.to_datetime(period_end))
        )
        df.loc[mask, pressure_column] = 0
        df.loc[mask, debit_column] = 0

# Обнуляем значения для прогнозной группы
for well in wells_to_zero_forecast:
    mask_forecast = (
        (df['Well Name'] == well) & 
        (df['Дата'] >= forecast_start) & 
        (df['Дата'] <= forecast_end)
    )
    df.loc[mask_forecast, pressure_column] = 0
    df.loc[mask_forecast, debit_column] = 0

# Построение графиков
plt.figure(figsize=(15, 10))

# Графики для исторической группы
for well in wells_to_zero_history:
    well_data = df[df['Well Name'] == well]
    plt.plot(well_data['Дата'], well_data[pressure_column], 
             label=f'{well} - История', alpha=0.7)

# Графики для прогнозной группы
for well in wells_to_zero_forecast:
    well_data = df[df['Well Name'] == well]
    plt.plot(well_data['Дата'], well_data[pressure_column], 
             label=f'{well} - Прогноз', alpha=0.7, linestyle='--')

plt.xlabel('Дата')
plt.ylabel('Значение')
plt.title('График обнуленных значений для всех скважин')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Сохранение графика и результатов
plt.savefig('zeroed_values_plot.png', dpi=300, bbox_inches='tight')
df.to_csv(data_path, index=False)
plt.show()
