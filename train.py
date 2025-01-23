# Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Embedding
from torch_geometric.nn import TransformerConv
from einops.layers.torch import Rearrange
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.engines import Predictor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tsl.nn.layers import NodeEmbedding, DiffConv
from tsl.nn.blocks.encoders import RNN
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Embedding
from torch_geometric.nn import TransformerConv
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt

class MultiWellTimeSeriesDataset: 
    def __init__(self, data, target_feature, well_column='Well Name', date_column='Дата'):
        self.data = data
        self.target_feature = target_feature
        self.well_column = well_column
        self.date_column = date_column
        self.n_nodes = data[well_column].nunique()

        # Modified date parsing with format='mixed' and dayfirst=True
        self.data[self.date_column] = pd.to_datetime(
            self.data[self.date_column], 
            format='mixed', 
            dayfirst=True   
        )
        self.well_coordinates = data.groupby(well_column)[['Initial X', 'Initial Y']].mean()

      

    def process_matrix_data(self, matrix_str):
        matrix = np.array(eval(matrix_str))
        return matrix.flatten()

    def get_connectivity(self, threshold=50, include_self=False, normalize_axis=1, layout="edge_index", use_weights=True):
        n_nodes = self.well_coordinates.shape[0]
        wells = self.well_coordinates.index
        distances = np.zeros((n_nodes, n_nodes))

        # Compute distances
        for i, well_i in enumerate(wells):
            for j, well_j in enumerate(wells):
                if i != j or include_self:
                    coord_i = self.well_coordinates.loc[well_i]
                    coord_j = self.well_coordinates.loc[well_j]
                    distances[i, j] = np.sqrt((coord_i['Initial X'] - coord_j['Initial X'])**2 +
                                            (coord_i['Initial Y'] - coord_j['Initial Y'])**2)
                else:
                    distances[i, j] = np.inf

        distances[distances == 0] = np.inf
        connectivity = (distances <= threshold).astype(int)

        if use_weights:
            with np.errstate(divide='ignore', invalid='ignore'):
                edge_weights = 1 / distances
                edge_weights[distances > threshold] = 0
                edge_weights[np.isinf(edge_weights)] = np.finfo(float).max

        if layout == "edge_index":
            edge_index = np.argwhere(connectivity)
            weights = edge_weights[edge_index[:, 0], edge_index[:, 1]] if use_weights else np.ones(edge_index.shape[0])
            return torch.tensor(edge_index.T, dtype=torch.int64), torch.tensor(weights, dtype=torch.float32)
        elif layout == "adjacency":
            return connectivity if not use_weights else (connectivity, edge_weights)
    def get_target_dataframe(self):
        target_df = self.data.pivot(index=self.date_column, columns=self.well_column, values=self.target_feature)
        target_df.fillna(0, inplace=True)
        return target_df

    def infer_mask(self, infer_from='next'):
        df = self.get_target_dataframe()
        mask = (~df.isna()).astype('uint8')
        eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns, data=0).astype('uint8')
        
        if infer_from == 'previous':
            offset = -1
        elif infer_from == 'next':
            offset = 1
        else:
            raise ValueError(f'`infer_from` can only be one of {["previous", "next"]}')

        months = sorted(set(zip(mask.index.year, mask.index.month)))

        for i in range(len(months)):
            j = (i + offset) % len(months)
            year_i, year_j = months[i][0], months[j][0]
            month_i, month_j = months[i][1], months[j][1]

            if pd.isnull(year_i) or pd.isnull(year_j) or pd.isnull(month_i) or pd.isnull(month_j):
                continue

            cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
            mask_j = mask[cond_j]
            offset_i = 12 * (year_i - year_j) + (month_i - month_j)

            if pd.isnull(offset_i):
                continue

            mask_i = mask_j.shift(1, pd.DateOffset(months=int(offset_i)))
            mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
            mask_i = mask_i.reindex(mask.index, fill_value=0)
            eval_mask |= (~mask_i & mask).astype('uint8')

        return eval_mask

    def __repr__(self):
        return f"MultiWellTimeSeriesDataset(length={len(self.data)}, n_nodes={self.n_nodes}, n_channels={self.n_channels})"

class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon


class TimeThenSpaceModel_Transformer(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32, num_transformer_layers: int = 2,
                 transformer_ff_size: int = 128, gnn_kernel: int = 2):
        super(TimeThenSpaceModel_Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = Embedding(n_nodes, hidden_size)
        transformer_layer = TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=transformer_ff_size, dropout=0.1, batch_first=True
        )
        self.time_nn = TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.space_nn = TransformerConv(in_channels=hidden_size, out_channels=hidden_size, heads=4, concat=False)  # Используем TransformerConv
        self.decoder = nn.Linear(hidden_size * n_nodes, horizon * n_nodes * input_size)
        self.rearrange = Rearrange('b (t n f) -> b t n f', t=horizon, n=n_nodes, f=input_size)

    def forward(self, x, edge_index, edge_weight):
        # print(f"Input x shape: {x.shape}")  # [6, 3, 30, 1]
        x_enc = self.encoder(x)
        # print(f"Encoded x shape: {x_enc.shape}")  # [6, 3, 30, 32]

        # Создание и подгонка размеров узловых встраиваний
        node_indices = torch.arange(x.size(2), device=x.device)  # [30]
        node_emb = self.node_embeddings(node_indices)  # [30, 32]
        # Подгоняем размерности node_emb для совпадения с размерами x_enc
        node_emb = node_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, 30, 32]
        node_emb = node_emb.expand(x.size(0), x.size(1), -1, -1)  # [6, 3, 30, 32]
        # print(f"Node embeddings shape: {node_emb.shape}")

        x_emb = x_enc + node_emb
        # print(f"Combined embeddings shape: {x_emb.shape}")  # [6, 3, 30, 32]

        # Плоское педставление для Transformer
        x_emb_flat = x_emb.view(x.size(0), -1, x_emb.size(-1))  # [6, 90, 32]
        # print(f"Flattened input to Transformer shape: {x_emb_flat.shape}")

        h = self.time_nn(x_emb_flat)
        h = h.view(x.size(0), x.size(1), x.size(2), -1)
        # print(f"Output from Transformer shape: {h.shape}")  # [6, 3, 30, 32]

        if edge_index is not None:
            h = h.view(-1, h.size(-1))  # [batch_size*time_steps*nodes, features]
            z = self.space_nn(h, edge_index)
            z = z.view(x.size(0), x.size(1), x.size(2), -1)
        else:
            z = h  # Если нет структуры графа

        # Теперь z имеет размер [6, 3, 30, 32]
        z_flat = z.view(z.size(0), z.size(1) * z.size(2) * z.size(3))  # [6, 2880]
        # print(f"Flattened output for Decoder shape: {z_flat.shape}")  # [6, 2880]

        # Исправление: убедимся, что размерности перед декодером правильные
        z_flat_correct = z.view(z.size(0), -1)[:, :self.hidden_size * self.n_nodes]  # [6, 960]
        # print(f"Reshaped for Decoder shape: {z_flat_correct.shape}")  # [6, 960]

        x_out = self.decoder(z_flat_correct)
        # print(f"Output from Decoder shape: {x_out.shape}")  # [6, 360]

        x_horizon = self.rearrange(x_out)
        # print(f"Final output shape: {x_horizon.shape}")  # [6, 12, 30, 1]

        return x_horizon




     
    
def create_pipeline():
    # Load and prepare data
    gdm_name = 'FY-SF-KM-1-1'
    data_path = f'train/{gdm_name}_merged_well_data.csv'
    data = pd.read_csv(data_path)
    dataset = MultiWellTimeSeriesDataset(data, target_feature='Дебит нефти (И), ст.бр/сут', date_column='Дата')
    
    # Generate required components
    target_df = dataset.get_target_dataframe()
    connectivity, weights = dataset.get_connectivity(threshold=50, include_self=False, normalize_axis=1, use_weights=True)
    edge_index, edge_weights = connectivity, weights
    mask = dataset.infer_mask()

    # Create TSL dataset
    torch_dataset = SpatioTemporalDataset(
        target=target_df.values,
        connectivity=(connectivity, weights),
        mask=mask.values,
        horizon=12,
        window=3,
        stride=1,
    )

    # Setup data module
    scalers = {'target': StandardScaler(axis=(0, 1))}
    splitter = TemporalSplitter(val_len=0.0, test_len=0.99)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=32
        )

    
    # Add this line to setup the datamodule
    dm.setup()
    input_size = torch_dataset.n_channels  
    n_nodes = torch_dataset.n_nodes         
    horizon = torch_dataset.horizon         

    # model = TimeThenSpaceModel(input_size=input_size,
    #                        n_nodes=n_nodes,
    #                        horizon=horizon,
    #                        hidden_size=32,
    #                        rnn_layers=1,
    #                        gnn_kernel=2)
  
    hidden_size = 32
    num_transformer_layers = 2
    transformer_ff_size = 128
    gnn_kernel = 2
    model = TimeThenSpaceModel_Transformer(input_size, n_nodes, horizon, hidden_size, num_transformer_layers, transformer_ff_size, gnn_kernel)
    # model = TimeThenSpaceModel(input_size=input_size,
    #                        n_nodes=n_nodes,
    #                        horizon=horizon,
    #                        hidden_size=32,
    #                        rnn_layers=1,
    #                        gnn_kernel=2)
    
    # Loss function and metrics
    loss_fn = MaskedMAE()
    metrics = {
        'mae': MaskedMAE(),
        'mape': MaskedMAPE(),
        'mae_next_year': MaskedMAE(at=12),
        'mae_in_2_months': MaskedMAE(at=2),
    }

    # Setup predictor
    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        loss_fn=loss_fn,
        metrics=metrics
    )

    return predictor, dm

if __name__ == "__main__":
    # Создание и обучение модели
    predictor, datamodule = create_pipeline()
    print("Model architecture:")
    print(predictor.model)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='logs',
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=500,
        # accelerator="cpu",  # Uncomment if you want to force CPU
        limit_train_batches=100,  # End an epoch after 100 updates
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(predictor, datamodule=datamodule)
