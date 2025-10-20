from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Concatenate, Input, Lambda


if __name__ == "__main__":
    # Load models
    model_brand = load_model('models/brand/vin_brand_predictor.keras')
    model_model = load_model('models/model/vin_model_predictor.keras')
    model_year = load_model('models/year/vin_year_predictor.keras')

    model_brand.trainable = False
    model_year.trainable = False
    model_model.trainable = False

    model_brand.name = "brand"
    model_model.name = "model"
    model_year.name = "year"

    # Create input
    input_shape = model_brand.input_shape[1:]
    joint_input = Input(shape=input_shape)

    # Get outputs
    output_brand = model_brand(joint_input)
    output_model = model_model(joint_input)
    output_year = model_year(joint_input)

    # Concatenate
    concatenated = Concatenate()([output_brand, output_model, output_year])

    # Create model
    joint_model = Model(inputs=joint_input, outputs=concatenated)

    joint_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Joint Model Summary:")
    joint_model.summary()

    # Save joint model
    joint_model.save('models/joint/joint_model.keras')
