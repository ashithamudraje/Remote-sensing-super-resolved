import tensorflow as tf
import torch
import numpy as np
from torchvision.models import resnet152
from torchvision.models import resnet50
from torchvision.models import resnet101
# Step 1: Load the TensorFlow ResNet152 model and checkpoint
def load_tf_model(checkpoint_dir, checkpoint_name):
    # Define your TensorFlow model architecture
    model = tf.keras.applications.ResNet101(weights=None)
    
    # Create a checkpoint object to restore the model's weights
    checkpoint = tf.train.Checkpoint(model=model)
    
    # Load the specific checkpoint file
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    
    # Check if the checkpoint file exists and restore the model
    if tf.io.gfile.exists(checkpoint_path + ".index"):
        print(f"Restoring model from checkpoint: {checkpoint_path}")
        checkpoint.restore(checkpoint_path).expect_partial()
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")
    
    return model


# Step 2: Extract weights from TensorFlow model
def extract_tf_weights(tf_model):
    tf_weights = {}
    for layer in tf_model.layers:
        layer_weights = layer.get_weights()  # This returns a list of weights for that layer
        if layer_weights:
            tf_weights[layer.name] = layer_weights
    return tf_weights


# Step 3: Define a PyTorch ResNet152 model
def load_torch_model():
    # Load pre-built PyTorch ResNet152 model
    torch_model = resnet101(pretrained=False)
    return torch_model


# Step 4: Convert TensorFlow weights to PyTorch format and transfer them
def convert_tf_to_torch(tf_weights, torch_model):
    state_dict = torch_model.state_dict()
    
    # Iterate over the PyTorch model's parameters
    for name, param in state_dict.items():
        # Assume the layer names in PyTorch and TensorFlow match well
        if name in tf_weights:
            tf_weight = np.array(tf_weights[name])

            # Handle weight reshaping: HWC (TensorFlow) to CHW (PyTorch) for Conv layers
            if len(tf_weight.shape) == 4:  # Conv layers
                tf_weight = np.transpose(tf_weight, (3, 2, 0, 1))  # HWC -> CHW

            # Assign the converted TensorFlow weight to the PyTorch parameter
            param.data = torch.tensor(tf_weight, dtype=param.dtype)

    return torch_model


# Step 5: Save the PyTorch model
def save_pytorch_model(torch_model, output_path):
    torch.save(torch_model.state_dict(), output_path)
    print(f"PyTorch model saved to {output_path}")


if __name__ == "__main__":
    # Path to the TensorFlow checkpoint directory
    tf_checkpoint_dir = "/netscratch/mudraje/super_resolution_remote_sensing/checkpoints/ResNet101_srresnet_32batch_normalization/models"
    
    # Name of the specific checkpoint file (without extension)
    checkpoint_name = "iteration-117992"
    
    # Step 1: Load the TensorFlow model and checkpoint
    tf_model = load_tf_model(tf_checkpoint_dir, checkpoint_name)
    
    # Step 2: Extract the weights from the TensorFlow model
    tf_weights = extract_tf_weights(tf_model)
    
    # Step 3: Load the PyTorch ResNet152 model
    torch_model = load_torch_model()
    
    # Step 4: Convert TensorFlow weights to PyTorch and transfer them
    converted_torch_model = convert_tf_to_torch(tf_weights, torch_model)
    
    # Step 5: Save the PyTorch model with the converted weights
    output_path = "/netscratch/mudraje/super_resolution_remote_sensing/converted_checkpoints/converted_Resnet101_srresnet_32batch_normalization.pth"
    save_pytorch_model(converted_torch_model, output_path) 
