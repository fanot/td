import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Predictor
from train import create_pipeline  

def evaluate_model():
    # Create model and data module
    predictor, dm = create_pipeline()
    
    # Load the best model weights
    checkpoint_path = 'logs/epoch=462-step=926.ckpt'  # Adjust path as needed
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
        
        x_with_pressure = x.clone()  
        
        # Setting zero pressure for half of the wells
        num_nodes = x_with_pressure.shape[2]
        zero_pressure_wells = list(range(num_nodes//2))
        x_with_pressure[:, :, zero_pressure_wells] = 0  # Зануляем значения для половины скважин
        
        with torch.no_grad():
            y_hat = predictor.model(x_with_pressure, edge_index, edge_weight)
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

def plot_predictions_combined(predictions_rescaled, targets_rescaled, node_index=11):
    stride = 1
    
    # Получаем полные предсказания
    full_predictions = predictions_rescaled[0]
    full_targets = targets_rescaled[0]
    
    for i in range(1, len(predictions_rescaled)):
        full_predictions = np.concatenate((full_predictions, predictions_rescaled[i][-stride:, :, :]), axis=0)
        full_targets = np.concatenate((full_targets, targets_rescaled[i][-stride:, :, :]), axis=0)
    
    time_steps = np.arange(full_predictions.shape[0])
    
    # Создаем график с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Верхний график - исходные значения
    ax1.plot(time_steps, full_predictions[:, node_index, 0], 'r--', label='Predicted Rate')
    ax1.plot(time_steps, full_targets[:, node_index, 0], 'b-', label='Actual Rate')
    ax1.set_title(f'Rate for Well {node_index+1}')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Rate')
    ax1.legend()
    
    # Нижний график - значения с нулевым давлением для половины скважин
    if node_index < full_predictions.shape[1] // 2:
        # Для скважин с нулевым давлением
        ax2.plot(time_steps, np.zeros_like(time_steps), 'b-', label='Pressure')
    else:
        # Для скважин с ненулевым давлением
        ax2.plot(time_steps, full_predictions[:, node_index, 0], 'r--', label='Normal Pressure (Predicted)')
        ax2.plot(time_steps, full_targets[:, node_index, 0], 'b-', label='Normal Pressure (Actual)')
    
    ax2.set_title(f'Pressure for Well {node_index+1}')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Pressure')
    ax2.legend()
    
    plt.tight_layout()
    
    # Сначала сохраняем
    plt.savefig(f'well_{node_index+1}_prediction.png')
    
    # Затем показываем
    plt.show()
    
    # И закрываем фигуру
    plt.close()

if __name__ == "__main__":
    predictions_rescaled, targets_rescaled = evaluate_model()
    
    # Строим графики для конкретной скважины
    plot_predictions_combined(predictions_rescaled, targets_rescaled)
    
    # Можно построить графики для других скважин
    # plot_predictions_combined(predictions_rescaled, targets_rescaled, node_index=29)