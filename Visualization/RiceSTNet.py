import torch
import torch.nn as nn

# Time2Vec Implementation
class Time2Vec(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="sin"):
        """
        Time2Vec layer using PyTorch's nn.Linear.
        Args:
        - in_features (int): The input feature size (usually 1 for time).
        - out_features (int): The total output size (D + 1), where (D = periodic components).
        - function (callable): The periodic activation function (default: sine).
        """
        super().__init__()
        self.periodic = nn.Linear(in_features, out_features-1, bias=True)  # Sinusoidal transformation
        self.linear = nn.Linear(in_features, 1, bias=True)  # Linear transformation
        self.function = torch.sin if activation == "sin" else torch.cos  # Can be sin or cos

    def forward(self, x):
        v1 = self.function(self.periodic(x))  # Apply sine/cosine to periodic part
        # print(v1.shape)
        v2 = self.linear(x)  # Linear transformation
        # print(v2.shape)
        return torch.cat([v1, v2], dim=-1)  # Concatenate along the last dimension

# CNN Model Definition
class CNNpart(nn.Module):
    def __init__(self):
        super().__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)  # 64 channels * 20x20 after pooling
        # self.relu_fc1 = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten the tensor
        x = self.flatten(x)  # Flatten starting from dimension 1
        # x = x.view(x.size(0), -1)  # Flatten to [batch_size, num_features]
        # Apply fully connected layers
        x = self.fc1(x)
        # x = self.relu_fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x

class GRUpart(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
          super().__init__()
          self.hidden_size = hidden_size
          self.n_layers = n_layers
          self.n_directions = 2 if bidirectional else 1
          # The inputs of GRU Layer with shape:
          # input: (batchSize, seqLen, input_size)
          # hidden: (nLayers * nDirections, batchSize, hiddenSize)
          # The outputs of GRU Layer with shape:
          # output: (batchSize, seqLen, hiddenSize * nDirections)
          # hidden: (nLayers * nDirections, batchSize, hiddenSize)
          self.gru = torch.nn.GRU(input_size, hidden_size, n_layers,
                                  bidirectional=bidirectional, batch_first=True)
          # input: (batchSize, hiddenSize*nDirections)
          # hidden: (batchSize, output_size)
          self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        if self.n_directions == 2:
            # hidden[0]: First layer (forward direction).
            # hidden[1]: First layer (backward direction).
            # hidden[2]: Second layer (forward direction).
            # hidden[3]: Second layer (backward direction)
            # Each has the shape: (batchSize, hiddenSize)
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

class PredictionModel(torch.nn.Module):
    def __init__(self, hidden_size, gruoutput_size, image_dim=128, time_dim=8):
        super().__init__()
        self.image_dim = image_dim
        self.time_dim = time_dim
        self.cnn = CNNpart()
        self.time2vec = Time2Vec(1, time_dim)
        self.gru = GRUpart(input_size = image_dim + time_dim,
                           hidden_size = hidden_size,
                           output_size = gruoutput_size)
        self.fc = nn.Linear(gruoutput_size, 1)  # Final output for prediction

    def forward(self, images, time_encodings):
        batch_size, seq_len, channels, height, width = images.shape

        # Process images through CNN: Extract spatial features
        images = images.view(batch_size * seq_len, channels, height, width)  # Reshape for CNN
        spatial_features = self.cnn(images)  # (Batch_Size * Seq_Len, 128)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)  # (Batch_Size, Seq_Len, 128)

        # Process time encoding through Time2Vec
        time_features = self.time2vec(time_encodings.view(batch_size * seq_len, 1))  # (Batch_Size * Seq_Len, Time_Dim)
        time_features = time_features.view(batch_size, seq_len, -1)  # (Batch_Size, Seq_Len, Time_Dim)

        # Concatenate spatial and time features
        combined_input = torch.cat([spatial_features, time_features], dim=-1)  # (Batch_Size, Seq_Len, 128 + Time_Dim)

        # Process sequence with GRU
        output = self.gru(combined_input)  # (Batch_Size, gruOutput_Size)
        output = self.fc(output) # (Batch_Size, 1)

        return output
