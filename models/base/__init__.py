import torch
from torchinfo import summary


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def info(self, input_size=(1, 5, 256, 256), depth=3):
        """Print model summary.

        Args:
            input_size (tuple): input size
        """
        return summary(self, input_size, depth=depth)

    def save(self, path):
        """Save model to file.

        Args:
            path (str): file path
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

        return self
