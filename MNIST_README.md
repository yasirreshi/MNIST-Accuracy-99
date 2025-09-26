# MNIST Digit Classification with a Lightweight Convolutional Neural Network

## 1. Project Overview and Goals

This project demonstrates the end-to-end process of building, training, and evaluating a lightweight Convolutional Neural Network (CNN) for classifying handwritten digits from the famous MNIST dataset. The entire implementation is done using the PyTorch deep learning framework.

The notebook is structured to follow industry best practices, including clear goal definition, hyperparameter configuration, detailed architectural explanations, data visualization, and model error analysis.

### Project Goals

The primary objectives for this project are:
1.  **High Accuracy:** Achieve a classification accuracy of over **99.4%** on the unseen test dataset.
2.  **Lightweight Model:** Keep the total number of trainable model parameters under **20,000**.
3.  **Efficient Training:** Reach the target accuracy in **15 epochs or less**.

These goals reflect a common real-world challenge: creating highly accurate models that are also efficient and small enough to be deployed on resource-constrained environments.

## 2. Configuration and Hyperparameters

This section centralizes all key settings for the experiment, which is crucial for reproducibility and tuning.

*   **`EPOCHS = 15`**: The model is trained for 15 full passes over the training dataset.
*   **`BATCH_SIZE = 128`**: The model processes 128 images at a time before updating its weights.
*   **`LEARNING_RATE = 0.01`** and **`MOMENTUM = 0.9`**: These are parameters for the `SGD` optimizer. The learning rate controls the step size of weight updates, while momentum helps the optimizer to accelerate in the correct direction and overcome local minima.
*   **`USE_CUDA` and `DEVICE`**: Automatically detects if a CUDA-enabled GPU is available and sets it as the active device for faster tensor computations.
*   **`torch.manual_seed(1)`**: Sets a seed for random number generation in PyTorch. This ensures that weight initializations and data shuffling are the same every time the code is run, leading to reproducible results.

## 3. Data Loading and Preprocessing

This section prepares the MNIST dataset for training and testing.

*   **`datasets.MNIST`**: This `torchvision` utility downloads (if needed) and loads the dataset.
*   **Transforms**: A sequence of operations is applied to each image:
    *   `transforms.ToTensor()`: Converts the image from a PIL Image format (range [0, 255]) to a PyTorch Tensor (range [0.0, 1.0]).
    *   `transforms.Normalize((0.1307,), (0.3081,))`: Normalizes the tensor's pixel values. The provided numbers are the global mean and standard deviation of the MNIST dataset. This centers the data around zero, which helps the model train more stably and efficiently.
*   **`DataLoader`**: This utility wraps the dataset for efficient iteration.
    *   `batch_size=128`: Delivers data in batches.
    *   `shuffle=True`: Randomizes the order of images at the start of each epoch. This is critical for preventing the model from learning spurious patterns related to data order.
    *   `num_workers=1`, `pin_memory=True`: Optimizations for faster data transfer from CPU to GPU when `USE_CUDA` is true.

## 4. Model Architecture Deep Dive

The model is a custom CNN designed for a balance of accuracy and parameter efficiency. It is built from sequential blocks, each with a specific role in feature extraction and dimensionality reduction.

**Architectural Breakdown:**

1.  **Input Block (`convblock1`):**
    *   `Conv2d(1 -> 16)`: Takes the 1-channel (grayscale) 28x28 image and produces 16 feature maps. The 3x3 kernel with no padding reduces the spatial dimension to 26x26.
    *   `BatchNorm2d(16)`: Normalizes the activations to stabilize training.
    *   `ReLU()`: Introduces non-linearity.
    *   `Dropout(0.05)`: Provides light regularization.

2.  **Convolution Block 1 (`convblock2`):**
    *   `Conv2d(16 -> 32)`: Increases channel depth to 32 to learn more complex features. The dimension shrinks to 24x24.

3.  **Transition Block (`pool1`, `convblock3`):**
    *   `MaxPool2d(2, 2)`: Halves the spatial dimensions from 24x24 to 12x12. This aggressively reduces computation and parameters while retaining the most prominent features.
    *   `Conv2d(32 -> 16)` using a **1x1 kernel**: This is a "bottleneck" layer. It reduces the channel depth from 32 to 16 without changing the spatial dimensions. This is a key technique for parameter efficiency.

4.  **Convolution Block 2 (`convblock4`, `convblock5`):**
    *   Two successive 3x3 convolutions that progressively extract more abstract features. The spatial dimensions are reduced from 12x12 -> 10x10 -> 8x8.

5.  **Output Block (`gap`, `convblock6`):**
    *   **`AdaptiveAvgPool2d((1, 1))` (GAP):** This is the final feature-pooling step. It takes the 8x8 feature maps and reduces each one to a single value (its average). This is a powerful alternative to a traditional `Flatten` and `Linear` layer, as it drastically reduces parameters and is less prone to overfitting.
    *   **`Conv2d(32 -> 10)` using a 1x1 kernel:** This acts as a fully connected layer. It maps the 32 channels from the GAP layer to the 10 output classes (digits 0-9).

**Final Activation (`forward` method):**

*   The `forward` method concludes with `F.log_softmax(x, dim=-1)`. This is a critical choice that pairs with the `F.nll_loss` (Negative Log Likelihood Loss) function used during training. The model outputs the *logarithm of probabilities* for each class.

## 5. Training and Evaluation

*   **Optimizer:** We use `optim.SGD` (Stochastic Gradient Descent) with a learning rate of `0.01` and momentum of `0.9`. SGD is a robust and well-understood optimizer, and momentum helps it navigate the loss landscape more effectively.
*   **Loss Function:** `F.nll_loss` is used. This is the standard choice when the model's final layer is `log_softmax`. It computes the loss based on the log-probability of the correct class.
*   **`train()` function:**
    1.  `model.train()`: Sets the model to training mode, enabling Dropout and ensuring BatchNorm uses batch statistics.
    2.  `optimizer.zero_grad()`: Clears gradients from the previous step.
    3.  `loss.backward()`: Computes gradients for all model parameters.
    4.  `optimizer.step()`: Updates the model's weights using the computed gradients.
*   **`test()` function:**
    1.  `model.eval()`: Sets the model to evaluation mode, disabling Dropout and making BatchNorm use its learned running statistics.
    2.  `with torch.no_grad()`: Disables gradient calculation to speed up inference and save memory.
    3.  The loss is aggregated using `reduction='sum'` and then averaged over the entire dataset for a stable metric.

## 6. Results and Analysis

The notebook includes functions to visualize the training and test accuracy/loss curves, which helps in diagnosing issues like overfitting. Additionally, it plots images that the model misclassified, allowing for qualitative error analysis to understand the model's weaknesses (e.g., confusion between digits like '4' and '9').