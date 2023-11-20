import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


'''
BEOFE YOU RUN THE CODE, make sure you have the following directories in your current directory:
    - checkpoints
    - trained_model
    - logs/fit
'''


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "trained_model"
checkpoint_path = "checkpoints/" + timestamp + "/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "logs/fit/" + timestamp

'''
After check the image dimension distribution, we need to resize and normalize the image for trianing purpose.
Theare are many ways to resize the image, I did some research on some releated research papers.
Reference paper:
MobileNetV2:Inverted Residuals and Linear Bottlenecks
Following the paper suggestion, I will resize the impage as the paper did, which is 224 x 224
'''
INPUT_SHAPE = (224, 224, 3)

# Information from the dataset description
NUM_BREEDS = 120


def load_dataset():
    '''
    Tensorfolw standfor_dogs dataset info:
    - Link: https://www.tensorflow.org/datasets/catalog/stanford_dogs
    - The dataset contains images of 120 breeds of dogs from around the world.
    - Total images in this dataset: 20,580
      - Train: 12,000
      - Test: 8580
    - Class labels and bounding box annotations ae provided fo r all 12,000 (train data) images

    Feature structure info for tfds:
        FeaturesDict({
            'image': Image(shape=(None, None, 3), dtype=uint8),
            'image/filename': Text(shape=(), dtype=string),
            'label': ClassLabel(shape=(), dtype=int64, num_classes=120),
            'objects': Sequence({
                'bbox': BBoxFeature(shape=(4,), dtype=float32),
            }),
        })
    '''
    (train, test), data_info = tfds.load('stanford_dogs', split=['train', 'test'], shuffle_files=True, with_info=True, data_dir='./tfds_dogs_dataset')

    return train, test, data_info


def get_image_data_info(dataset, dataset_flag):
    image_width, image_height = [], []
    for data in dataset:
        width = data['image'].shape[0]
        height = data['image'].shape[1]
        image_width.append(width)
        image_height.append(height)
    
    # visualize the image dimension distribution
    plt.subplot(211)
    plt.hist(x=image_width, bins=20, alpha=0.7, density=True)
    # plt.axvline(max(set(img_width), key=img_width.count), color='r')
    plt.ylabel('Pixels')
    plt.ylabel('Frequency')
    plt.title('Image width distribution')
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(212)
    plt.hist(x=image_height, bins=20, alpha=0.7)
    # plt.axvline(max(set(img_height), key=img_height.count), color='r')
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    plt.title('Image height distribution')
    plt.grid(axis='y', alpha=0.75)

    plt.subplots_adjust(hspace=0.5)
    # plt.show()       
    plt.savefig('{}_data_width_height_distribution.png'.format(dataset_flag)) 


def image_preprocess(image_data, image_size, num_labels):
    '''
    After check the image dimension distribution, we need to resize and normalize the image for trianing purpose.
    Theare are many ways to resize the image, I did some research on some releated research papers.
    Reference paper:
    MobileNetV2:Inverted Residuals and Linear Bottlenecks
    Following the paper suggestion, I will resize the impage as the paper did, which is 224 x 224
    '''
    image = image_data['image']
    label = image_data['label']
    # casting image to float32
    image = tf.cast(image, tf.float32) 
    image = tf.image.resize(image, image_size, method='nearest')
    image = image / 255.
    label = tf.one_hot(label, num_labels)

    return image, label


def data_pipeline(dataset, image_shape, num_classes, batch_size=None):
    dataset = dataset.map(lambda x: image_preprocess(x, image_shape[0:-1], num_classes),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if batch_size:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def cnn_model(image_shape, num_classes, lr=0.001):
    model = Sequential([
        Conv2D(16, 3, activation='relu', use_bias=False, padding='same', input_shape=image_shape),
        MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
        # Dropout(rate=0.2),
        Conv2D(32, 3, activation='relu', use_bias=False, padding='same'),
        MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
        Dropout(rate=0.2),
        Conv2D(64, 3, activation='relu', use_bias=False, padding='same'),
        Conv2D(64, 3, activation='relu', use_bias=False, padding='same'),
        MaxPool2D(pool_size=(3, 3), strides=2),
        # Dropout(rate=0.2),
        Flatten(),
        Dense(128, activation='relu'),
        # Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


def cnn_model2(image_shape, num_classes, lr=0.001):
    '''
    NOTE: this model using MobileNetV2 model from tentorflow. This model has better perforamance with shorter training
    time, but make sure this is valid implementation in your project.
    '''
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def model_train(model, train_batches, test_batches, epochs):
    print("Training model ...")
    print(model.summary())
   
 
    # Create a callback for visualization in TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          profile_batch=0)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Train
    model.fit(
        train_batches,
        epochs=epochs,
        validation_data=test_batches,
        callbacks=[tensorboard_callback, cp_callback]
    )

    # save the model weights
    model.save(save_path + "/stanford_dog_cnn")


def model_evaluate(model, test_batches):
    print("Evaluating trained model...")
    metrics = model.evaluate(test_batches, return_dict=True, verbose=1)
    for key, val in metrics.items():
        print(key + ": {:.2f}".format(val))


def main():
    train_data, test_data, data_info = load_dataset()
    # get_image_data_info(train_data, 'train')
    # get_image_data_info(test_data, 'test')
    # train_batches = data_pipeline(train_data, INPUT_SHAPE, NUM_BREEDS, batch_size=32)
    # test_batches = data_pipeline(test_data, INPUT_SHAPE, NUM_BREEDS,  batch_size=32) 
    
    # EPOCHS = 10
    # tf.random.set_seed(123)
    # model = cnn_model(INPUT_SHAPE, NUM_BREEDS, lr=0.001)
    # model = cnn_model2(INPUT_SHAPE, NUM_BREEDS, lr=0.0001)
    # model_train(model, train_batches, test_batches, epochs=EPOCHS)

#    model_evaluate(model, test_batches)    


if __name__ == "__main__":
    main()

