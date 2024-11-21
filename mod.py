import pandas as pd
import ast
import random

def modify_well_history(df):
    # Получаем уникальные имена скважин
    wells = df['Well Name'].unique()
    
    # Выбираем случайную половину скважин
    wells_to_modify = random.sample(list(wells), len(wells)//2)
    
    # Для каждой строки в датафрейме
    for idx, row in df.iterrows():
        if row['Well Name'] in wells_to_modify:
            # Для каждой колонки, содержащей списки
            for col in ['Дебит нефти (И), ст.бр/сут', 'Дебит воды (И), ст.бр/сут', 
                       'Дебит жидкости (И), ст.бр/сут', 'Забойное давление (И), Фунт-сила / кв.дюйм (абс.)',
                       'WBP9, Фунт-сила / кв.дюйм (абс.)']:
                if col in df.columns:
                    try:
                        # Преобразуем строку в список
                        values = ast.literal_eval(row[col])
                        # Заменяем последнюю треть значений нулями
                        cut_point = len(values) * 2 // 3
                        values[cut_point:] = [0.0] * (len(values) - cut_point)
                        # Обновляем значение в датафрейме
                        df.at[idx, col] = str(values)
                    except:
                        continue
    
    return df

# Чтение файла
df = pd.read_csv('merged_well_data.csv')

# Модификация данных
modified_df = modify_well_history(df)

# Сохранение результата
modified_df.to_csv('modified_merged_well_data.csv', index=False)