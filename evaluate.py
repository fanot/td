import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Predictor
from train import create_pipeline  
import pandas as pd
from datetime import timedelta  
import datetime
from tqdm import tqdm


def evaluate_model():
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
    
    # Get validation dataloader
    test_dataloader = dm.val_dataloader()
   
    # Collect predictions
    predictions = []
    targets = []
    
    for batch in test_dataloader:
        x = batch.input['x']  # Input data [b, t=3, n, f]
        y = batch.target['y']  # Target data [b, t=12, n, f]
        edge_index = batch.input['edge_index']
        edge_weight = batch.input['edge_weight']
        
        print(f"Input tensor shape: {x.shape}")
        
        # x_with_pressure = x.clone()  
        
        # # Setting zero pressure for half of the wells
        # num_nodes = x_with_pressure.shape[2]
        # zero_pressure_wells = list(range(num_nodes//2))
        # x_with_pressure[:, :, zero_pressure_wells] = 0  # Зануляем значения для половины скважин
        
        with torch.no_grad():
            y_hat = predictor.model(x, edge_index, edge_weight)
        predictions.append(y_hat.cpu().numpy())
        targets.append(y.cpu().numpy())
    
    # Convert lists to NumPy arrays
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Reverse scaling
    scaler = dm.scalers['target']
    scale = scaler.scale.cpu().numpy()
    bias = scaler.bias.cpu().numpy()
    
    predictions_rescaled = predictions * scale + bias
    targets_rescaled = targets
    
    return predictions_rescaled, targets_rescaled

def plot_predictions_for_all_wells(predictions_rescaled, targets_rescaled, num_nodes):
    stride = 1

    for node_index in range(num_nodes):
        full_predictions = predictions_rescaled[0]
        full_targets = targets_rescaled[0]

        # Получаем полные предсказания
        for i in range(1, len(predictions_rescaled)):
            full_predictions = np.concatenate((full_predictions, predictions_rescaled[i][-stride:, :, :]), axis=0)
            full_targets = np.concatenate((full_targets, targets_rescaled[i][-stride:, :, :]), axis=0)

        # Находим индекс, где заканчиваются начальные нули
        start_idx = 0
        while start_idx < len(full_predictions) and full_predictions[start_idx, node_index+2, 0] <= 0:
            start_idx += 1

        # Обрезаем начальные нули
        full_predictions = full_predictions[start_idx:]
        full_targets = full_targets[start_idx:]

        time_steps = np.arange(full_predictions.shape[0])

        # Загрузка данных из CSV файла
        data = pd.read_csv('modified_merged_well_data.csv')

        # Преобразование столбца 'Дата' в формат datetime
        data['Дата'] = pd.to_datetime(data['Дата'], format='%Y-%m-%d')

        # Фильтрация данных по одной скважине
        well_id = f'P{node_index+1}'
        filtered_data = data[
            (data['Well Name'] == well_id) & 
            (
                ((data['Дата'].dt.year == 2009) & (data['Дата'].dt.month >= 4)) |  
                (data['Дата'].dt.year > 2009)  
            )
        ].copy()

        # Создаем график с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        start_date = datetime.datetime(2009, 4, 1)  # Начальная дата 
        dates = [start_date + datetime.timedelta(days=30 * i) for i in range(len(time_steps))]

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
        plt.savefig(f'FY-SF-KM-12-12/FY-SF-KP-5-31_{node_index+1}_prediction.png')

        # Затем показываем
        plt.show()

        # И закрываем фигуру
        plt.close()

        results_df = pd.DataFrame({
            'Date': dates,
            'Model': ['FY-SF-KM-12-12'] * len(dates),
            'Well': [f'Well_{node_index+1}'] * len(dates),
            'Predicted_Rate': full_predictions[:, node_index+2, 0],
            'Actual_Rate': full_targets[:, node_index+2, 0],
            'Bottom_Hole_Pressure': pressure_data
        })

        # Сохраняем результаты в CSV
        results_df.to_csv(f'FY-SF-KM-12-12/well_{node_index+1}_predictions.csv', index=False)

if __name__ == "__main__":
    predictions_rescaled, targets_rescaled = evaluate_model()
    
    # Определяем количество узлов (скважин)
    num_nodes = predictions_rescaled.shape[2]
    
    # Строим графики для всех скважин
    plot_predictions_for_all_wells(predictions_rescaled, targets_rescaled, num_nodes)
