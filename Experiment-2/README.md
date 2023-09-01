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

### Applying the GradCAM instance over the images

The below instructions run on a loop for each image present in the `img_list`

- The `input image` is converted into a `floating point` format and normalizes the data, dividing it by `255`. This will make the elements present in the `input image tensor` to vary between `0 and 1`. The result is stored inside the `input_tensor`
- The `input_tensor` is modified by applying `preprocessing` steps. These steps include `resizing the image` to a specific size; `normalizing the image` this normalization is done by subtracting the mean and dividing by the standard deviation of the images's pixel values, this is specifically done to ensure that pixel values have a standard scale and are centered around `zero`
- The instantiated `cam` gradcam object and the `input_tensor` is used to extract the `grayscal_cam` which identifies the `regions` on which our Neural Network model sees that part as the important area for the prediction. The `targets` parameter specifies the `target class` for which we want to compute the GradCAM heatmap. `targets` is set to `None`, to visualize the heatmap without focusing on a specific class. If we want to highlight the regions important for a specific class, we would specify the `class index`.