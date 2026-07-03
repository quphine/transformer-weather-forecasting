# Project Overview

This project features a hand-built transformer implementation from scratch. Core features implemented manually:

- Sinusoidal Encodings
- Multi-Head Attention class
- Embedding Model class
- Encoder Blocks class
- Custom Dataset Class


## Implementation Summary


|         |                                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Task**               | Direct Multi-Step Weather Forecasting                                                                                               |
| **Framework**          | PyTorch                                                                                                                             |
| **Architecture**       | Transformer Encoder + Conv1D Downsampler                                                                                                               |
| **Input Window**       | 4320 timesteps                                                                                                                       |
| **Output Horizon**     | 144 timesteps                                                                                                                       |
| **Forecasting Type**   | Non-Autoregressive                                                                                                                  |
| **Dataset**            |Jena Climate dataset                                                                                            |
| **Features**           | Temperature, Humidity, Pressure, Wind Speed, Wind Direction, etc.                                                                   |
| **Core Components**    | Linear Projection, Sinusoidal Positional Encoding, Multi-Head Self-Attention, Feed-Forward Network, LayerNorm, Residual Connections |
| **Loss Function**      | Mean Squared Error (MSE)                                                                                                            |
| **Evaluation Metrics** | MAE, RMSE, R² Score                                                                                                                 
| **Objective**          | Learn long-range temporal dependencies for efficient direct multi-horizon forecasting.                                              |

## Model Architecture
```mermaid
flowchart TD
    A["Input<br/>(B, 4320, 14)"]

    B["Linear Projection<br/>14 → 128"]

    C["Transpose<br/>(B, 128, 4320)"]

    D["Conv1D + GELU<br/>Kernel = 5<br/>Stride = 4<br/>Output: (B, 128, 1080)"]

    E["Conv1D + GELU<br/>Kernel = 5<br/>Stride = 2<br/>Output: (B, 128, 540)"]

    F["Transpose Back<br/>(B, 540, 128)"]

    G["Add Positional Encoding"]

    H["Encoder Block ×3<br/>Pre-LayerNorm<br/>Multi-Head Attention<br/>Feed Forward Network"]

    I["Last 3 Tokens<br/>(B, 3, 128)"]

    J["Flatten<br/>(B, 384)"]

    K["LayerNorm"]

    L["Linear<br/>384 → 144"]

    M["Forecast Output<br/>(B, 144)"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
```




### Compact stage table

| Stage | Operation | Shape before | Shape after |
|---|---|---|---|
| Raw input | — | `(B, 4320, 14)` | `(B, 4320, 14)` |
| Input proj | `Linear(14,128)` | `(B, 4320, 14)` | `(B, 4320, 128)` |
| Prep for conv | `transpose(1,2)` | `(B, 4320, 128)` | `(B, 128, 4320)` |
| Conv1 | `Conv1d(..., stride=4)` | `(B, 128, 4320)` | `(B, 128, 1080)` |
| Conv2 | `Conv1d(..., stride=2)` | `(B, 128, 1080)` | `(B, 128, 540)` |
| Back to seq | `transpose(1,2)` | `(B, 128, 540)` | `(B, 540, 128)` |
| MHA internals | Split heads / attention / merge | `(B, 540, 128)` | `(B, 540, 128)` |
| Encoder stack | 3 × (Pre-LN MHA + FFN) | `(B, 540, 128)` | `(B, 540, 128)` |
| Readout | `x[:, -3:, :]` | `(B, 540, 128)` | `(B, 3, 128)` |
| Flatten | `Flatten()` | `(B, 3, 128)` | `(B, 384)` |
| Decoder | `LayerNorm(384) -> Linear(384,144)` | `(B, 384)` | `(B, 144)` |

---
## Results

### Overall Performance

| **Metric**              |    **Value** |
| ----------------------- | -----------: |
| MAE                     | **1.825 °C** |
| RMSE                    | **2.417 °C** |
| R² Score                |   **0.8914** |
| Correlation Coefficient |   **0.9446** |

### Error Distribution

| **Statistic** |    **Value** |
| ------------- | -----------: |
| Best MAE      | **0.382 °C** |
| Median MAE    | **1.597 °C** |
| Mean MAE      | **1.825 °C** |
| Worst MAE     | **7.228 °C** |

### Error Analysis

| **MAE Threshold**      | **Samples** |
| ---------------------- | ----------: |
| > 1 °C                 |        1305 |
| > 2 °C                 |         514 |
| > 3 °C                 |         175 |
| > 4 °C                 |          49 |
| > 5 °C                 |          13 |
| > 6 °C                 |           4 |
| **Total Test Samples** |    **1548** |
