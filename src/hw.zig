const std = @import("std");
const io = std.io;

/// Function: updateWeights
/// Description: Update the weights of the model using gradient descent algorithm.
/// Input: weights - array of current weights, gradients - gradient values for corresponding weights, learningRate - rate of learning.
/// Output: Mutates the weights array in place by adjusting it based on gradients.
pub fn updateWeights(weights: []f32, gradients: []f32, learningRate: f32) void {
    // Loop through each weight and update it based on gradient descent formula.
    for (weights.len) |i| {
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
    var rng = std.rand.DefaultPrng.init(1234); // Seed-based pseudo-random number generator.
    var allocator = std.heap.page_allocator;

    // Allocate memory for gradients.
    var gradients = try allocator.alloc(f32, initialParams.len);
    defer allocator.free(gradients);

    for (0..epochs) |epoch| {
        // Shuffle the dataset for stochastic gradient descent.
        for (data) |sample| {
            const rand_index = rng.random.usize(0, data.len);

            // Calculate gradients for the current sample.
            computeGradient(data[rand_index], gradients, params);

            // Update parameters based on the calculated gradients.
            updateWeights(params, gradients, learningRate);
        }
    }

    return params;
}

/// Function: main
/// Description: Main entry point of the program. Initializes parameters and executes the SGD algorithm.
/// Calls readData to read the dataset from file, and finally prints the optimized parameters.
pub fn main() !void {
    const data = try readData("grad.txt"); // Call the function to read the dataset.

    var allocator = std.heap.page_allocator;

    var initialParams: [2]f32 = [2]f32{ 0.0, 0.0 }; // Initialize model parameters.
    var initialParamsSlice: []f32 = initialParams[0..];

    const learningRate: f32 = 0.01; // Set learning rate.
    const epochs: u32 = 1000; // Set number of training epochs.

    var finalParams = try SGD(data, initialParamsSlice, learningRate, epochs); // Execute SGD.
    std.debug.print("Final parameters: {}\n", .{finalParams}); // Output the final optimized parameters.
}

/// Function: readData
/// Description: Reads data from a text file and returns it as a 2D array of floats.
/// Input: filename - string containing the name of the text file.
/// Output: 2D array of floats containing the dataset.
fn readData(filename: []const u8) ![][]f32 {
    var file = try std.fs.cwd().openFile(filename, std.fs.File.OpenFlag.read);
    defer file.close();

    var allocator = std.heap.page_allocator;

    // Allocate memory for the outer array.
    var data = try allocator.alloc([]f32, 100); // Assuming max 100 rows for simplicity.
    var count: usize = 0; // To keep track of number of rows in the data array.

    const buffer_size = 1024; // Size of the static buffer.
    const buffer = [_]u8{0} ** buffer_size; // Declare a static buffer which will be used by the BufferedReader to store read data temporarily.
    
    // Create a BufferedReader to efficiently read the file. The BufferedReader uses the static buffer we provided.
    var reader = io.bufferedReader(file.reader()).reader();

    // Continuously read each line from the file until EOF is encountered. Assumes that each line ends with '\n'.
    while (true) {
        const line = try reader.readUntilDelimiterOrEof('\n');
        if (line == null) break; // When readUntilDelimiterOrEof returns null, it signals EOF, so break the while loop.

        // line is a slice pointing to the part of the buffer containing the line read from the file (excluding the newline character if any).
        // Process the line here (e.g., split the line on commas, parse each field).

        // Example of processing the line (not robust, for simple parsing demonstration)
        var cells = std.mem.tokenize(line.?, ",");
        var data_row = try allocator.alloc(f32, 4); // example: assuming each line will have 4 float numbers
        defer allocator.free(data_row); // Defer the deallocation of data_row to when it goes out of scope.
        var index: usize = 0;
        for (cells) |cell| {
            data_row[index] = try std.fmt.parseFloat(f32, cell); // Convert each cell into a floating-point number and store it in data_row.
            index += 1;
        }
        data[count] = data_row; // Store the parsed row in the data array.
        count += 1;
    }

    return data[0..count];
}
