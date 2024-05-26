## Stochastic Gradient Descent (SGD) for Linear Regression

This project implements a simple Linear Regression optimization using the Stochastic Gradient Descent (SGD) algorithm. The main goal is to update model parameters (weights) to minimize prediction error based on a given dataset.

### Features

- **updateWeights**: Updates model weights using gradient descent.
- **computeGradient**: Computes gradients based on the prediction error.
- **SGD**: Performs Stochastic Gradient Descent to optimize model parameters.
- **readData**: Reads dataset from a text file.
- **main**: Main entry point which initializes parameters, reads the dataset, and performs SGD.

### Files

- `grad.txt`: Text file containing the dataset used for training.
- `sgd.zig`: Main script implementing the SGD algorithm.
- `compile_time.sh`: Script to calculate compile time.

### Installation and Setup

1. **Install Zig [accordingly](https://ziglang.org/download/)**
  
2. **Clone repository and compile program:**

   ```bash
   git clone https://github.com/paxamans/zig_project
   cd zig_project
   cd src
   zig build-exe sgd.zig
   ```
3. **Run program if no errors ecountered:**

   ```bash
   ./sgd.zig
   ```
   
### Deprecation
This project will no longer be maintained due to a lack of motivation. Feel free to fork and continue development.
