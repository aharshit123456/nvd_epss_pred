# Description: Utility functions for the training and evaluation of the model.


# Freeze most layers of the model
def freeze_model_layers(model, num_layers_to_freeze=None):
    """
    Freeze the layers of the model.
    If num_layers_to_freeze is None, freeze all layers except the last few.
    """
    # Inspect the model's architecture
    for name, module in model.named_children():
        print(f"Layer: {name}, Type: {type(module)}")

    # Find transformer blocks dynamically
    transformer_blocks = None
    for name, module in model.named_children():
        if "transformer" in name.lower():
            transformer_blocks = module
            break

    if transformer_blocks is None:
        raise ValueError("Cannot find transformer blocks in the model.")

    # Dynamically access the layers within the transformer
    transformer_layers = list(transformer_blocks.children())
    total_layers = len(transformer_layers)
    print(total_layers)
    if num_layers_to_freeze is None:
        num_layers_to_freeze = total_layers - 2  # Default: Freeze all but the last 2 layers

    # Freeze specified number of layers
    for i, layer in enumerate(transformer_layers[:num_layers_to_freeze]):
        for param in layer.parameters():
            param.requires_grad = False

    print(f"Froze {num_layers_to_freeze} out of {total_layers} transformer layers.")
