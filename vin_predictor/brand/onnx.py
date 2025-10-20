import tf2onnx

if __name__ == "__main__":
    tf2onnx.convert.from_keras(
        "vin_brand_predictor.keras",
        opset=13,
        output_path="vin_brand_95.onnx"
    )