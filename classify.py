import tensorflow as tf
class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, weights, gradients):
        return [w - (self.learning_rate * g) if g is not None else w for w, g in zip(weights, gradients)]
class NeuralNetwork:
    def __init__(self, num_layers, num_nodes_per_layer, learning_rate):
        self.num_layers = num_layers
        self.num_nodes_per_layer = num_nodes_per_layer

        # Initialize weights and biases with He initialization
        self.weights = [tf.Variable(tf.keras.initializers.HeNormal()(shape=(num_nodes_per_layer[i], num_nodes_per_layer[i+1])))
                        for i in range(num_layers)]
        self.biases = [tf.Variable(tf.zeros(shape=(num_nodes_per_layer[i+1], 1))) for i in range(num_layers)]

        # Define optimizer
        self.optimizer = GradientDescent(learning_rate)
    def forward(self, X):
        Y_pred = X
        for l in range(self.num_layers):
            Y_pred = tf.matmul(Y_pred, self.weights[l]) + tf.transpose(self.biases[l])
            Y_pred = tf.nn.relu(Y_pred)  # ReLU activation
        return Y_pred

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                Y_pred = self.forward(X_train)
                loss = tf.reduce_mean(tf.square(Y_train - Y_pred))

            # Calculate gradients
            gradients = tape.gradient(loss, self.weights + self.biases)

            # Update weights and biases using GradientDescent class
            self.weights = self.optimizer.update(self.weights, gradients[:self.num_layers])
            self.biases = self.optimizer.update(self.biases, gradients[self.num_layers:])
            # Print loss every epoch
            print("Epoch:", epoch, "Loss:", loss.numpy())
nn = NeuralNetwork(2, [4, 2, 2, 2], 0.01)

# Generate some training data (replace with your actual data)
X_train = tf.random.normal([100, 4])
Y_train = tf.random.normal([100, 2])

# Train the neural network for 10 epochs
nn.train(X_train, Y_train, 10)
#preprocessing ,data agumentation,clssification,normalization,batch normali