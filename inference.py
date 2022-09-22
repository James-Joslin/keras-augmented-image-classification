import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy

import external as ext

HEIGHT = 75
WIDTH = 75
DEPTH = 3
BATCH_SIZE = 32
EPOCHS = 250
SEED = 7

image_data = ext.getImageData(
    train_dir = "./train", val_dir = "./val", test_dir = "./test"
).loadImageData(
    HEIGHT, WIDTH, DEPTH, BATCH_SIZE, SEED
)
INDICES = image_data['test'].class_indices
print(INDICES)

model = ext.load_model(
    save_dir = "./", model_name = "multiclass_image_classification"
)
model.compile(
    optimizer = "Nadam",
    loss = SparseCategoricalCrossentropy(
        from_logits=False),
    metrics='Accuracy'
)

model.evaluate(image_data['test'])

# Run inference on selected files

