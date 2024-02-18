# Self-attention-and-Convolution-Model
* This work can be found in  [[Renewable Energy]](https://www.sciencedirect.com/science/article/abs/pii/S0960148123013149)).

📍 This work proposes a wind power prediction model, in which the proposed model uses self-attention to capture the long-range relationship and uses convolutional layers to learn the local temporal interactions of the variables in the time-series data.

📍 Compared to deep learning sequence models, such as recurrent neural network (RNN), long short-term memory (LSTM), and gated recurrent unit (GRU), the proposed method can simultaneously consider global and local information.


# Model

# Training
## Setting
The data used in the wind power prediction task belong to time-series data, which means that a series of data points are collected over time. We introduce the notation for time series data at different time steps as ***x1, x2, . . . , xT*** to denote time series data of length ***T***, each time steps has ***D*** dimensions.


<img src="pic/corerlation.png">
