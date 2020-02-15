import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.input_dim = config['input_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.batch_first = config['batch_first']
        self.batch_size = config['batch_size']
        self.dropout = config['dropout']
        self.ouput_size = config['output_size']
        self.verbose = config['verbose']


        # LSTM Layer
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        #self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)  # TODO needed?

        # Dense Output layer w/o activation
        self.linear = nn.Linear(self.hidden_dim, self.ouput_size)

    def forward(self, input):

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        lstm1_out, _ = self.lstm1(input)

        #lstm2_out, _ = self.lstm2(lstm1_out)

        if self.verbose:
            print('lstm out shape')
            print(lstm1_out.shape)
        y_pred = self.linear(lstm1_out)
        if self.verbose:
            print('y_pred shape')
            print(y_pred.shape)
        self.verbose = False

        return y_pred  # of shape (batch_size, seq len, out size)


class FCNN(nn.Module):

    def __init__(self, config):
        super(FCNN, self).__init__()

        self.input_dim = config['input_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.batch_first = config['batch_first']
        self.batch_size = config['batch_size']
        self.dropout = config['dropout']
        self.ouput_size = config['output_size']
        self.verbose = config['verbose']

        # Input layer activation
        self.in_layer = nn.Sequential(
            nn.Linear(self.input_dim, out_features=150),
            nn.ReLU())
        self.layer1 = nn.Sequential(
            nn.Linear(150, out_features=50),
            nn.ReLU())
        # Dense Output layer w/o activation
        self.linear = nn.Linear(50, self.ouput_size)

    def forward(self, input):

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        out = self.in_layer(input)
        out = self.layer1(out)
        y_pred = self.linear(out)

        # lstm2_out, _ = self.lstm2(lstm1_out)

        return y_pred  # of shape (batch_size, seq len, out size)


class ConvSkip(nn.Module):

    def __init__(self, config):
        super(ConvSkip, self).__init__()

        self.input_dim = config['input_size']
        self.dropout = config['dropout']
        self.output_size = config['output_size']

        # Input layer activation
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))

        self.regression_part = nn.Sequential(
            nn.Linear(3072, out_features=1536),
            nn.ReLU(),
            nn.Linear(1536, out_features=512),
            nn.ReLU(),
            nn.Linear(512, out_features=self.output_size))

    def forward(self, input):

        # Now it's tricky: we pick the two images separately, and we add a fake number of channels (1)
        img_1 = self.conv_1(input[:, 0, None, :, :])
        img_2 = self.conv_1(input[:, 1, None, :, :])

        img_1 = self.conv_2(img_1)
        img_2 = self.conv_2(img_2)

        img_1 = self.conv_3(img_1)
        img_2 = self.conv_3(img_2)

        img_1 = img_1.view(img_1.size(0), -1)
        img_2 = img_2.view(img_2.size(0), -1)

        img_stacked = torch.cat((img_1, img_2), 1)

        y_pred = self.regression_part(img_stacked)

        return y_pred  # of shape (batch_size, len, out size)
