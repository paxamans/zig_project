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
fn computeGradient(sample: []const f32, gradients: []f32, weights: []const f32) void {
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
pub fn SGD(data: [][]const f32, initialParams: []f32, learningRate: f32, epochs: u32) ![]f32 {
    const params = initialParams; // Using `const` since it is not re-assigned.
    var rng = std.rand.DefaultPrng.init(1234); // Seed-based pseudo-random number generator.
    var allocator = std.heap.page_allocator;

    // Allocate memory for gradients.
    const param_len = initialParams.len;
    const gradients = try allocator.alloc(f32, param_len);
    defer allocator.free(gradients);

    for (0..epochs) |_| { // `|_|` for unused epoch capture.
        for (data) |_| { // `|sample|` to correctly handle sample iteration.
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
    const allocator = std.heap.page_allocator;
    var mutable_allocator = allocator; // Create a mutable copy of the allocator
    const data = try readData("grad.txt", &mutable_allocator);
    defer for (data) |*row| mutable_allocator.free(row); // Free each allocated row in data.

    const initialParams: [2]f32 = [2]f32{ 0.0, 0.0 }; // Initialize model parameters.
    const initialParamsSlice: []f32 = initialParams[0..];

    const learningRate: f32 = 0.01; // Set learning rate.
    const epochs: u32 = 1000; // Set number of training epochs.

    const finalParams = try SGD(data, initialParamsSlice, learningRate, epochs); // Execute SGD.
    std.debug.print("Final parameters: {}\n", .{finalParams}); // Output the final optimized parameters.
}

/// Function: readData
/// Description: Reads data from a text file and returns it as a 2D array of floats.
/// Input: filename - string containing the name of the text file.
/// Output: 2D array of floats containing the dataset.
fn readData(filename: []const u8, allocator: *std.mem.Allocator) ![][]f32 {
    const file = try std.fs.cwd().openFile(filename, .{ .read = true });
    defer file.close();

    var data = std.ArrayList([]f32).init(allocator);

    // BufferedReader for efficient file reading.
    var reader = io.BufferedReader.init(file.reader());

    // Read each line from the file until EOF.
    while (true) {
        const line = try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 1024);
        defer allocator.free(line.buf); // Free the line buffer after processing.
        if (line.buf.len == 0) break; // EOF check.

        // Split the line into tokens based on commas.
        const tokens = std.mem.split(line.buf, ",");
        var data_row = try allocator.alloc(f32, tokens.len); // Allocate a row for the parsed float data.

        // Convert each token into a floating-point number.
        var index: usize = 0;
        for (tokens) |token| {
            data_row[index] = try std.fmt.parseFloat(f32, token);
            index += 1;
        }

        // Append the parsed row to the data array.
        try data.append(data_row);
    }

    // Return the data as a slice.
    return data.toOwnedSlice();
}
