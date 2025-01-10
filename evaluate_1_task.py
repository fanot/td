import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Predictor
from train import create_pipeline  
import pandas as pd
from datetime import timedelta  
import datetime
from tqdm import tqdm


def create_batch_from_input(current_input):
    """
    Создает новый батч данных из текущего ввода.
    Предполагается, что current_input - это словарь с ключами: 'x', 'edge_index', 'edge_weight'.
    """
    return {
        'input': {
            'x': current_input['x'],
            'edge_index': current_input['edge_index'],
            'edge_weight': current_input['edge_weight']
        }
    }

def update_input_with_prediction(current_input, y_hat):
    """
    Обновляет current_input новыми предсказаниями.
    Предполагается, что обновление осуществляется только для 'x'.
    """
    # Клонируем текущий ввод, чтобы не изменять оригинал
    updated_input = {}
    for k, v in current_input.items():
       if not isinstance(v, torch.Tensor):
           v = torch.tensor(v)  # Convert to tensor if not already
       updated_input[k] = v.clone()
    
    
    updated_input['x'][:] = y_hat[:, :updated_input['x'].shape[1], ...]  # Adjust slicing as needed

    return updated_input
def evaluate_model_for_2020():
    # Create model and data module
    predictor, dm = create_pipeline()

    # Load the best model weights
    checkpoint_path = 'logs/epoch=496-step=1491.ckpt'  
    predictor.load_model(checkpoint_path)

    # Verify model loading
    if predictor.model is None:
        raise Exception("Model was not loaded correctly.")

    # Set model to evaluation mode
    predictor.model.eval()

    # Prepare input data for initial prediction
    start_date = pd.to_datetime('2019-12-31')  # Initial point is the last day of 2019
    prediction_dates = pd.date_range(start_date, periods=1, freq='D')

    # Get initial input using the last data available
    test_dataloader = dm.val_dataloader()

    # Collect predictions for 2020
    predictions = []
    current_input = None

    for step in tqdm(range(365)):  # 365 days in 2020
        if step == 0:
            # Use the real data from the last available point
            batch = next(iter(test_dataloader))
        else:
            # Create new batch from current_input
            batch = create_batch_from_input(current_input)
        print(batch)
        x = batch['x']
        edge_index = batch['edge_index']
        edge_weight = batch['edge_weight']

        with torch.no_grad():
            y_hat = predictor.model(x, edge_index, edge_weight)
            predictions.append(y_hat.cpu().numpy())

        # Update current_input with the new prediction
        current_input = update_input_with_prediction(batch, y_hat)

    # Convert predictions to NumPy array
    predictions = np.concatenate(predictions, axis=0)

    # Reverse scaling
    scaler = dm.scalers['target']
    scale = scaler.scale.cpu().numpy()
    bias = scaler.bias.cpu().numpy()

    predictions_rescaled = predictions * scale + bias

    return predictions_rescaled

def plot_predictions_combined(predictions_rescaled, targets_rescaled, node_index=16):
    node_index = node_index - 1
    stride = 1
    full_predictions = predictions_rescaled[0]
    full_targets = targets_rescaled[0]

    # Получаем полные предсказания
    for i in range(1, len(predictions_rescaled)):
        # Добавляем только неперекрывающуюся часть каждого батча
        full_predictions = np.concatenate((full_predictions, predictions_rescaled[i][-stride:, :, :]), axis=0)
        full_targets = np.concatenate((full_targets, targets_rescaled[i][-stride:, :, :]), axis=0)

    # Находим индекс, где заканчиваются начальные нули
    start_idx = 0
    while start_idx < len(full_predictions) and full_predictions[start_idx, node_index+2, 0] <= 0:
        start_idx += 1

    # # Обрезаем начальные нули
    # full_predictions = full_predictions[start_idx:]
    # full_targets = full_targets[start_idx:]
    time_steps = np.arange(full_predictions.shape[0])
    print(time_steps)
  
    # Загрузка данных из CSV файла
    data = pd.read_csv('modified_merged_well_data.csv')

    # Преобразование столбца 'Дата' в формат datetime
    data['Дата'] = pd.to_datetime(data['Дата'], format='%Y-%m-%d')

    # Фильтрация данных по одной скважине
    well_id = 'P16'
    filtered_data = data[
        (data['Well Name'] == well_id) & 
        (
            ((data['Дата'].dt.year == 2009) & (data['Дата'].dt.month >= 4)) |  #
            (data['Дата'].dt.year > 2009)  
        )
    ].copy()


    # Создаем график с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    start_date = datetime.datetime(2009, 4, 1) # Начальная дата 
    dates = [start_date +  datetime.timedelta(days=30 * i) for i in range(len(time_steps))]
    
    
    # Верхний график - исходные значения
    ax1.plot(dates, full_predictions[:, node_index+2, 0], 'r--', label='Predicted Values')
    ax1.plot(dates, full_targets[:, node_index+2, 0], 'b-', label='Actual Values')
    
    ax1.set_title(f'Rate for Well {node_index+1} на модели FY-SF-KM-12-12')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Rate')
    ax1.legend()
    ax1.grid(True)
        # Обрезаем или дополняем данные забойного давления, чтобы длины совпадали
    pressure_data = filtered_data['Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'].values
    if len(pressure_data) > len(dates):
        pressure_data = pressure_data[:len(dates)]
    elif len(pressure_data) < len(dates):
        # Дополняем нулями или последним значением, если нужно
        padding = np.full(len(dates) - len(pressure_data), pressure_data[-1])
        pressure_data = np.concatenate([pressure_data, padding])
    
    # Нижний график - забойное давление
    ax2.plot(dates, pressure_data, label=f'Забойное давление {well_id}')
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Забойное давление (Фунт-сила / кв.дюйм)')
    ax2.set_title(f'График забойного давления для скважины')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Сначала сохраняем
    plt.savefig(f'well_{node_index+1}_prediction.png')
    
    # Затем показываем
    plt.show()
    
    # И закрываем фигуру
    plt.close()
    # results_df = pd.DataFrame({
    #         'Date': dates,
    #         'Model': ['FY-SS-KS-13-65'] * len(dates),
    #         'Well': [f'Well_{node_index+1}'] * len(dates),
    #         'Predicted_Rate': full_predictions[:, node_index+2, 0],
    #         'Actual_Rate': full_targets[:, node_index+2, 0],
    #         'Bottom_Hole_Pressure': pressure_data
    #     })
        
    #     # Сохраняем результаты в CSV
    # results_df.to_csv(f'well_{node_index+1}_predictions.csv', index=False)
        
if __name__ == "__main__":
    predictions_rescaled_2020 = evaluate_model_for_2020()

    # Строим графики для конкретной скважины с данными за 2020 год
    plot_predictions_combined(predictions_rescaled_2020, node_index=16)

    
    # Можно построить графики для других скважин
    # plot_predictions_combined(predictions_rescaled, targets_rescaled, node_index=29)