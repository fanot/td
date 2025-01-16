import pandas as pd
import matplotlib.pyplot as plt

# Читаем файлы
brugge = pd.read_csv('data/FY-SF-KM-12-12.csv', delimiter='\t')
merged = pd.read_csv('merged_well_data.csv')

# Преобразуем давление из строки в числовой формат
def convert_pressure(x):
    try:
        if isinstance(x, str):
            if x == 'Забойное давление (И), Бара':  # Пропускаем заголовок
                return None
            # Убираем '+' и конвертируем научную нотацию
            return float(x.replace('+', ''))
        elif pd.isna(x):
            return None
        else:
            return float(x)
    except:
        return None

# Применяем конвертацию
brugge['Забойное давление (И), Бара'] = brugge['Забойное давление (И), Бара'].apply(convert_pressure)

# Фильтруем только данные для P16 перед созданием словаря
brugge_p16 = brugge[brugge['Объект'] == 'P16']

# Создаем словари для маппинга значений по датам
oil_rate_dict = dict(zip(brugge_p16['Дата'], pd.to_numeric(brugge_p16['Дебит нефти, ст.м3/сут'], errors='coerce')))
bhp_dict = dict(zip(brugge_p16['Дата'], brugge_p16['Забойное давление (И), Бара']))

# Проверяем значения в словаре bhp_dict
print("\nSample values from bhp_dict after conversion:")
for date in list(bhp_dict.keys())[:5]:
    print(f"Date: {date}, Pressure: {bhp_dict[date]}")

# Обновляем значения в merged_data с конвертацией единиц
# Конвертируем из ст.м3/сут в ст.бр/сут (1 м3 = 6.289811 баррелей)
merged['Дебит нефти (И), ст.бр/сут'] = pd.to_numeric(merged['Дата'].map(oil_rate_dict), errors='coerce') * 6.289811

# Конвертируем из Бар в Фунт-сила/кв.дюйм (1 бар = 14.5038 psi)
merged['Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'] = pd.to_numeric(merged['Дата'].map(bhp_dict), errors='coerce') * 14.5038

# Проверяем наличие данных после преобразования
print("\nSample of merged data for P16:")
p16_merged = merged[merged['Well Name'] == 'P16']
print(p16_merged[['Дата', 'Забойное давление (И), Фунт-сила / кв.дюйм (абс.)']].head())
print("\nStatistics for P16 pressure:")
print(p16_merged['Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'].describe())

# Сохраняем обновленный файл
merged.to_csv('updated_merged_well_data.csv', index=False)

# Фильтруем данные только для скважины P16
merged_p16 = merged[merged['Well Name'] == 'P16']

# Создаем график
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# График дебита нефти
plot_data = merged_p16.dropna(subset=['Дата', 'Дебит нефти (И), ст.бр/сут'])
ax1.plot(pd.to_datetime(plot_data['Дата'], format='%d.%m.%Y'), 
         plot_data['Дебит нефти (И), ст.бр/сут'], 
         'b-', label='Дебит нефти P16')
ax1.set_title('Дебит нефти скважины P16')
ax1.set_xlabel('Дата')
ax1.set_ylabel('Дебит нефти, ст.бр/сут')
ax1.grid(True)
ax1.legend()

# График забойного давления
plot_data = merged_p16.dropna(subset=['Дата', 'Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'])
ax2.plot(pd.to_datetime(plot_data['Дата'], format='%d.%m.%Y'),
         plot_data['Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'], 
         'r-', label='Забойное давление P16')
ax2.set_title('Забойное давление скважины P16')
ax2.set_xlabel('Дата')
ax2.set_ylabel('Забойное давление, Фунт-сила/кв.дюйм')
ax2.grid(True)
ax2.legend()

# Настраиваем layout
plt.tight_layout()

# Сохраняем график
plt.savefig('P16_parameters.png')
plt.show() 