from torch import Tensor, abs


def loss(outputs: Tensor, targets: Tensor, no_data: float = 0) -> Tensor:
    """Computes the mean absolute error between the predicted and target values.

    Args:
        outputs (Tensor): The predicted values.
        targets (Tensor): The target values.
        no_data (float, optional): Specifies the no data value. Defaults to -np.inf.

    Returns:
        Tensor: The mean absolute error.
    """

    # Flatten arrays
    outputs = outputs.flatten()
    targets = targets.flatten()

    # Remove no data values
    outputs = outputs[targets != no_data]
    targets = targets[targets != no_data]

    return abs(targets - outputs).mean()
