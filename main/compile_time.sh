#!/bin/bash

# Start timing
start_time=$(date +%s%N) # Capture start time in nanoseconds

zig build-exe hw.zig

# End timing
end_time=$(date +%s%N) # Capture end time in nanoseconds
compile_time_ns=$((end_time - start_time)) # Calculate compile duration in nanoseconds
compile_time_ms=$((compile_time_ns / 1000000)) # Convert to milliseconds

# Print the compile time
echo "Compilation took $compile_time_ms milliseconds."
