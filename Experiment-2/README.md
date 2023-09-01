In this experiment, I have tried out the `pypi grad-cam` library to understand the GradCAM feature.

- GradCAM (Gradient-weighted Class Activation Mapping) is a technique used in deep learning and computer vision to visualize the regions of an input image that contribute the most to the predictions made by a neural network model. 
- It highlights the important regions by generating a heatmap that shows where the model's attention is focused within the image. 
- GradCAM achieves this by using gradients of the model's output with respect to the input image to determine the importance of each pixel.

### Function Definition

Let's understand how the function defined in the **GradCAM/Experiment-2/**`gradcam.py` works internally. This function accepts, the below parameters:

    - `model` - The `Neural Network model` for which GradCAM will be computed.
    - `target_layer` - The `layer` in the model to visualize.
    - `img_list` - A list of `images` to apply GradCAM to.
    - `preprocess_args` - A dictionary of preprocessing `arguments` for the images.
    - `**kwargs` - TODO

### GradCAM Instance creation

To instantiate the `GradCAM`, the `trained_model` and `target_layer` are used. This instance helps us visualize what parts of an image the neural network paid the most `attention` to, when making a particular `prediction`. It's a tool for interpreting why the network made a specific decision.
