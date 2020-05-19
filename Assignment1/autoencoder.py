import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder where the decoder structure simply mirrors the encoder structure
    """

    def __init__(self, n_input, h_units, non_lin):
        """
        Decoder is designed to model the
        :param n_input: no. of input units to the network
        :param h_units: list of size n_layers
        """
        super(Autoencoder, self).__init__()
        layer_list = []
        h_layers = len(h_units)

        def add_layer_and_act(n_inp, n_hidden, nl_type): return [nn.Linear(n_inp, n_hidden), self.non_lin(nl_type)]

        # Build Encoder Layers
        layer_list.extend(add_layer_and_act(n_input, h_units[0], non_lin))

        if len(h_units)>1:
            for i in range(1, h_layers):
                layer_list.extend(add_layer_and_act(h_units[i-1], h_units[i], non_lin))

        # Build Decoder Layers
        if len(h_units) > 1:
            for i in range(h_layers-1, 0, -1):
                layer_list.extend(add_layer_and_act(h_units[i], h_units[i-1], non_lin))

        # Build Output layer
        non_lin = 'sigmoid' # Output sigmoidal regardless of network non-linearities
        layer_list.extend(add_layer_and_act(h_units[0], n_input, non_lin))

        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)

    def non_lin(self, nl_type='sigmoid'):
        """
        Simply plugs in a predefined non-linearity from a dictionary to be used throughout the network
        :param nl_type: type based on predefined types. Defaults to sigmoid on wrong type.
        :return:
        """
        nl = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'relu': nn.ReLU()}
        try:
            return nl[nl_type]
        except:
            print("non linearity type not found. Defaulting to sigmoid.")
            return nl['sigmoid']


if __name__ == "__main__":

    model = Autoencoder(784, [20, 10], 'sigmoid')
    from torchsummary import summary
    summary(model, input_size=(1, 784))