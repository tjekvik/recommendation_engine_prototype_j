# Use official Python runtime as base image
FROM python:3.11-slim

# Create directory for volume mount
RUN mkdir -p /onnx

# Declare volume for read-write access
VOLUME ["/onnx"]

# Run the application
CMD ["sleep", "infinity"]