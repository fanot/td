import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Predictor
from train import create_pipeline  
import pandas as pd
from datetime import timedelta  
import datetime
from tqdm import tqdm


def evaluate_model(model):
    # Create model and data module
    predictor, dm = create_pipeline()
    
    # Load the best model weights
    checkpoint_path = model
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
        x = batch.input['x']  
        y = batch.target['y']  
        edge_index = batch.input['edge_index']
        edge_weight = batch.input['edge_weight']
        
        
   
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

def plot_predictions_for_all_wells(gdm_name, predictions_rescaled, targets_rescaled, num_nodes):
    stride = 1
    well_name = gdm_name
    
    # Загрузка данных из CSV файла
    data_path = f'train/{well_name}_merged_well_data.csv'
    data = pd.read_csv(data_path)
    
    # Преобразование столбца 'Дата' в формат datetime
    date_formats = ['%Y-%m-%d', '%d.%m.%Y']
    for fmt in date_formats:
        try:
            data['Дата'] = pd.to_datetime(data['Дата'], format=fmt)
            break
        except ValueError:
            continue
    
    # Получаем список скважин и сортируем их
    unique_wells = sorted(data['Well Name'].unique())
    # Разделяем на нагнетательные и добывающие
    inj_wells = sorted([w for w in unique_wells if w.startswith('I')])
    prod_wells = sorted([w for w in unique_wells if w.startswith('P')])
    
    # Объединяем списки в правильном порядке: сначала I, потом P
    all_wells = inj_wells + prod_wells
    
    print(f"Injection wells: {inj_wells}")
    print(f"Production wells: {prod_wells}")
    print(f"Total wells in order: {all_wells}")
    
    # Обрабатываем все скважины в порядке I, затем P
    for node_index, well_id in enumerate(all_wells):
        if node_index >= num_nodes:
            break
            
        print(f"Processing well: {well_id} (node_index: {node_index})")
        
        full_predictions = predictions_rescaled[0]
        full_targets = targets_rescaled[0]

        # Получаем полные предсказания
        for i in range(1, len(predictions_rescaled)):
            full_predictions = np.concatenate((full_predictions, predictions_rescaled[i][-stride:, :, :]), axis=0)
            full_targets = np.concatenate((full_targets, targets_rescaled[i][-stride:, :, :]), axis=0)

       
        time_steps = np.arange(full_predictions.shape[0])
        
        filtered_data = data[
            (data['Well Name'] == well_id) & 
            (
                ((data['Дата'].dt.year == 2009) & (data['Дата'].dt.month >= 4)) |  
                (data['Дата'].dt.year > 2009)  
            )
        ].copy()

        # Создаем график с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        start_date = datetime.datetime(2009, 4, 1)
        dates = [start_date + datetime.timedelta(days=30 * i) for i in range(len(time_steps))]

        # Верхний график - исходные значения
        ax1.plot(dates, full_predictions[:, node_index, 0], 'r--', label='Predicted Values')
        ax1.plot(dates, full_targets[:, node_index, 0], 'b-', label='Actual Values')

        ax1.set_title(f'Rate for Well {well_id}')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Rate')
        ax1.legend()
        ax1.grid(True)

        pressure_data = filtered_data['Забойное давление (И), Фунт-сила / кв.дюйм (абс.)'].values
        pressure_data = np.concatenate(([0], pressure_data))
        
        if len(pressure_data) > len(dates):
            pressure_data = pressure_data[:len(dates)]
        elif len(pressure_data) < len(dates):
            padding = np.full(len(dates) - len(pressure_data), pressure_data[-1])
            pressure_data = np.concatenate([pressure_data, padding])

        # Нижний график - забойное давление
        ax2.plot(dates, pressure_data, label=f'Забойное давление')
        ax2.set_xlabel('Дата')
        ax2.set_ylabel('Забойное давление (Фунт-сила / кв.дюйм)')
        ax2.set_title(f'График забойного давления для скважины {well_id}')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(f'{well_name}/well_{well_id}_prediction.png')
        plt.show()
        plt.close()

        # Сохраняем результаты в CSV
        results_df = pd.DataFrame({
            'Date': dates,
            'Model': [well_name] * len(dates),   
            'Well': [well_id] * len(dates),
            'Predicted_Rate': full_predictions[:, node_index, 0],
            'Actual_Rate': full_targets[:, node_index, 0],
            'Bottom_Hole_Pressure': pressure_data
        })
        
        results_df.to_csv(f'{well_name}/well_{well_id}_predictions.csv', index=False)

if __name__ == "__main__":
    model = 'logs/epoch=477-step=1434.ckpt'  
    gdm_name = 'FY-SF-KM-1-1'
    predictions_rescaled, targets_rescaled = evaluate_model(model)
    num_nodes = predictions_rescaled.shape[2] 

    # Строим графики для всех скважин
    plot_predictions_for_all_wells(gdm_name, predictions_rescaled, targets_rescaled, num_nodes)
