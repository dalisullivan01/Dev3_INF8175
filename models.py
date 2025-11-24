import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        return nn.DotProduct(self.w, x)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        # Tant qu'il y a des erreurs, recommencer
        done = False
        while not done:
            done = True  # deviendra False si une erreur est trouvée
            
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                real_y = nn.as_scalar(y)

                if prediction != real_y:
                    # Mise à jour perceptron : w ← w + y * x
                    direction = nn.Constant(real_y * x.data)
                    self.w.update(direction, 1)
                    
                    done = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.hidden_size = 50

        # Paramètres de la couche 1 (1 -> 50)
        self.W1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        # Paramètres de la couche 2 (50 -> 1)
        self.W2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        # Première couche : x → Linear + bias → ReLU
        h = nn.Linear(x, self.W1)
        h = nn.AddBias(h, self.b1)
        h = nn.ReLU(h)

        # Sortie : h → Linear + bias
        out = nn.Linear(h, self.W2)
        out = nn.AddBias(out, self.b2)

        return out

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        pred = self.run(x)
        return nn.SquareLoss(pred, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        batch_size = 100
        learning_rate = 0.01

        while True:
            total_loss = 0
            batches = 0

            for x, y in dataset.iterate_once(batch_size):
                # Loss du batch
                loss = self.get_loss(x, y)

                # Accumule pour calculer la loss moyenne
                total_loss += nn.as_scalar(loss)
                batches += 1

                # Calcul du gradient
                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])

                # Mise à jour des paramètres
                self.W1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.W2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)

            # Loss moyenne sur l'époque
            average_loss = total_loss / batches

            # Condition d'arrêt demandée dans l'énoncé
            if average_loss <= 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.hidden1 = 200
        self.hidden2 = 100

        self.W1 = nn.Parameter(784, self.hidden1)
        self.b1 = nn.Parameter(1, self.hidden1)

        self.W2 = nn.Parameter(self.hidden1, self.hidden2)
        self.b2 = nn.Parameter(1, self.hidden2)

        self.W3 = nn.Parameter(self.hidden2, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Layer 1
        h1 = nn.Linear(x, self.W1)
        h1 = nn.AddBias(h1, self.b1)
        h1 = nn.ReLU(h1)

        # Layer 2
        h2 = nn.Linear(h1, self.W2)
        h2 = nn.AddBias(h2, self.b2)
        h2 = nn.ReLU(h2)

        # Output layer (logits)
        out = nn.Linear(h2, self.W3)
        out = nn.AddBias(out, self.b3)

        return out

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        batch_size = 100
        learning_rate = 0.1

        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)

                gradients = nn.gradients(loss,
                    [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

                self.W1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.W2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)
                self.W3.update(gradients[4], -learning_rate)
                self.b3.update(gradients[5], -learning_rate)

            # Check validation accuracy
            if dataset.get_validation_accuracy() >= 0.97:
                break

