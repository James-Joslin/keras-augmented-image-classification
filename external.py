import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


class getImageData(object):

    def __init__(self, train_dir, val_dir, test_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            shear_range = 0.2,
            zoom_range = 0.2,
            fill_mode="reflect",
            brightness_range=(0.8, 1.5),
            horizontal_flip=True
        )

        self.val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        self.test_datagen = ImageDataGenerator(
            rescale=1./255
        )
    
    def __printClassNames (self, dataset):
        print(dataset.class_indices)
        for image_batch, labels_batch in dataset:
            print(f'Image batch shape: {image_batch.shape}')
            print(f'Image labels shape: {labels_batch.shape}\n')
            break
    
    def loadImageData (self, set_height, set_width, set_depth, batch, seed):
        color_modes = ["grayscale", "rgb", "rgba"]
        color = color_modes[set_depth-2]
        
        print("Training data:")
        train_data = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size = (set_height, set_width),
            class_mode='sparse',
            shuffle=True,
            color_mode=color,
            batch_size=batch,
            seed=seed
        )
        self.__printClassNames(train_data)

        print("Validation data:")
        val_data = self.val_datagen.flow_from_directory(
            self.val_dir,
            target_size = (set_height, set_width),
            class_mode='sparse',
            shuffle=False,
            color_mode=color,
            batch_size=batch,
            seed=seed
        )
        self.__printClassNames(val_data)

        print("Testing data:")
        test_data = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size = (set_height, set_width),
            class_mode='sparse',
            shuffle=False,
            color_mode=color,
            batch_size=batch,
            seed=seed
        )
        self.__printClassNames(test_data)

        data_dict = {
            "train": train_data,
            "val" : val_data,
            "test" : test_data
        }
        return data_dict

class customModel (object):
    def __init__(self, set_height, set_width, set_colour_depth) -> None:
        self.height = set_height
        self.width = set_width
        self.depth = set_colour_depth

    def buildModel (self, num_outputs, optimiser, loss_metric):
        model = Sequential(
            [
                layers.Input(shape=(self.height , self.width , self.depth)),
            
                #Block 1
                layers.Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                
                #Block 2
                layers.Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu'), 
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                
                #Block 3
                layers.Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu'),
                layers.MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same'),
                
                #Block 5
                layers.Flatten(),
                layers.Dense(units = 256 , activation='relu'),
                layers.Dropout(rate = 0.2),
                layers.Dense(units = 128 , activation='relu'),
                layers.Dropout(rate = 0.2),
                layers.Dense(units = num_outputs , activation='softmax')
            ]
        )
        model.compile(
            metrics=['Accuracy'],
            optimizer=optimiser,
            loss = loss_metric
        )
        model.summary()
        return model

def visualise_performance (history):
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history['loss'], label = "Loss")
    axs[0].plot(history.history['val_loss'], label = "Validation Loss")
    axs[0].legend()
    axs[1].plot(history.history['Accuracy'], label = "Accuracy")
    axs[1].plot(history.history['val_Accuracy'], label = "Validation Accuracy")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def save_model(model_in, save_dir = "", save_name = ""):
    model_json = model_in.to_json()
    with open(f'{os.path.join(save_dir,save_name)}.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    # serialize weights to HDF5
    model_in.save_weights(f'{os.path.join(save_dir,save_name)}.hdf5')
    print("Saved model to disk")

def load_model(save_dir = "", model_name = ""):
    print("Loading Precomputed Model")
    json_file = open(f'{os.path.join(save_dir,model_name)}.json', 'r').read()
    model = model_from_json(json_file)
    # load weights into new model
    model.load_weights(f'{os.path.join(save_dir,model_name)}.hdf5')
    print("Loaded model from disk")
    model.summary()
    return model

def model_accuracy(model, data):
    model.evaluate(data)
    y_pred = model.predict(data)

    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.constant(data.labels)

    labels = np.array(list(data.class_indices.keys()))
    print(labels)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=labels,
                yticklabels=labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Real Label')
    plt.show()

if __name__ == "__main__":

    pass