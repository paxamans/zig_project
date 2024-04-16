const std = @import("std");
const io = std.io;

/// Function: updateWeights
/// Description: Update the weights of the model using gradient descent algorithm.
/// Input: weights - array of current weights, gradients - gradient values for corresponding weights, learningRate - rate of learning.
/// Output: Mutates the weights array in place by adjusting it based on gradients.
pub fn updateWeights(weights: []f32, gradients: []f32, learningRate: f32) void {
    // Loop through each weight and update it based on gradient descent formula.
    for (weights, 0..) |_, i| {
        weights[i] -= learningRate * gradients[i];
    }
}

/// Function: computeGradient
/// Description: Computes gradients based on the current model prediction error.
/// Input: sample - single data sample (feature vector), gradients - array to hold computed gradients, weights - current model weights.
/// Output: Fills the gradients array with computed gradient values.
fn computeGradient(sample: []f32, gradients: []f32, weights: []f32) void {
    // Extract feature and label from sample.
    const x = sample[0];
    const y = sample[1];
    const predicted = weights[0] * x + weights[1]; // Calculate prediction using model.
    const diff = predicted - y; // Difference between prediction and actual label.
    // Compute gradients for linear regression.
    gradients[0] = 2 * diff * x;
    gradients[1] = 2 * diff;
}

/// Function: SGD
/// Description: Performs Stochastic Gradient Descent to optimize the model parameters.
/// Input: data - dataset, initialParams - initial model parameters, learningRate - learning rate, epochs - number of epochs.
/// Output: Returns the optimized model parameters.
/// Throws: Can throw an error if memory allocation for gradients fails.
pub fn SGD(data: [][]f32, initialParams: []f32, learningRate: f32, epochs: u32) ![]f32 {
    var params = initialParams; // Work with a mutable copy of initial parameters.
    var rng = std.rand.DefaultPrng.init(1234); // Seed-based pseudo-random generator for sampling.
    // Perform epoch iterations for stochastic gradient descent.
    for (0..epochs) |_| {
        var dataIndex = rng.random.uintLessThan(data.len); // Select a random index for stochastic nature.
        var sample = data[dataIndex];
        var gradients: []f32 = try std.heap.page_allocator.alloc(f32, params.len); // Allocate space for gradients.
        defer std.heap.page_allocator.free(gradients); // Automatic cleanup of gradients.
        computeGradient(sample, gradients, params); // Compute gradients for the selected sample.
        updateWeights(params, gradients, learningRate); // Update model parameters based on gradients.
    }
    return params; // Return the optimized parameters after all epochs.
}
/// Function: main
/// Description: Main entry point of the program that reads CSV data and processes it.
/// Throws: Any errors related to file handling or data parsing will cause the function to exit with an error state.
pub fn main() anyerror!void {
    const allocator = std.heap.page_allocator; // Use the page allocator from the standard library for any needed dynamic memory allocation.
    const allocator = std.heap.page_allocator;

    // Open the file 'data.csv' from the current working directory with read permissions
    const file = try std.fs.cwd().openFile("data.csv", .{ .read = true });
    defer file.close(); // Ensure the file is closed when the function exits, whether normally or due to an error.
    defer file.close();
    var buffer: [4096]u8 = undefined;
    var reader = io.bufferedReader(file.reader()).reader();

    // Initialize a dynamic array to store the data.
    var data = try allocator.alloc([]f32, 100); // Assuming a maximum of 100 samples for simplicity.
    var count: usize = 0;

    while (true) {
        const line = try reader.readUntilDelimiterOrEof('\n');
        if (line == null) break;
        // line is a slice pointing to the part of the buffer containing the line read from the file (excluding the newline character if any).
        var data_row = try allocator.alloc(f32, 2); // Assuming each line has 2 float numbers.
        var index: usize = 0;
        var index: usize = 0;
            data_row[index] = try std.fmt.parseFloat(f32, cell);
            index += 1;
        }
        data[count] = data_row; // Store the parsed row in the data array.
        count += 1;
    }
    var initialParams: [2]f32 = [0.0 0.0];
    var initialParamsSlice: []f32 = initialParams[0..];
    const learningRate: f32 = 0.01;
    const epochs: u32 = 1000;

    // Adjust the call to SGD to pass a slice of the data array up to the actual count of read samples.
    var finalParams = try SGD(data[0..count], initialParamsSlice, learningRate, epochs);
    std.debug.print("Final parameters: {}\n", .{finalParams});

    // Free the allocated memory for each data row and the data array itself.
    for (data[0..count]) |row| {
        allocator.free(row);
    }
    allocator.free(data);
}
