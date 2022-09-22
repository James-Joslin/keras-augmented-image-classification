import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy

import external as ext

if __name__ == "__main__":
    
    HEIGHT = 75
    WIDTH = 75
    DEPTH = 3

    model = ext.load_model(
        save_dir = "./saved_models/", model_name = "multiclass_image_classification"
    )
    model.compile(
        optimizer = "Nadam",
        loss = SparseCategoricalCrossentropy(
            from_logits=False),
        metrics='Accuracy'
    )
    output, probability = ext.run_inference(
        model = model,
        file = "data/inference/403.jpg",
        model_width = WIDTH,
        model_height = HEIGHT, 
        model_depth = DEPTH
    )
    print(f'Prediction: {output}\nProbability: {probability}')