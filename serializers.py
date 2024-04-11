import io

import torch
import torch.optim as optim
from model import BasicCNN
     
def serialize_model_weights(model):
    """Serialize PyTorch model weights to a bytes object."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()

def deserialize_model_weights(serialized_weights):
    """Deserialize bytes object back into PyTorch model weights."""
    buffer = io.BytesIO(serialized_weights)
    state_dict = torch.load(buffer, map_location='cpu')  # Adjust map_location as necessary
    return state_dict

def serialize_gradients(gradients):
    """Serialize a dictionary of gradient tensors to a bytes object."""
    buffer = io.BytesIO()
    # Ensure gradients are detached and on CPU for serialization
    gradients_cpu = {name: grad.detach().cpu() for name, grad in gradients.items() if grad is not None}
    torch.save(gradients_cpu, buffer)
    return buffer.getvalue()

def deserialize_gradients(serialized_gradients):
    """Deserialize bytes object back into a dictionary of gradient tensors."""
    buffer = io.BytesIO(serialized_gradients)
    gradients = torch.load(buffer)
    return gradients