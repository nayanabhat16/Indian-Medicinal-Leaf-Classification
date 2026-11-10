
import tensorflow as tf
from tensorflow.keras import layers, models


def build_densenet121(num_classes):

    base_model = tf.keras.applications.DenseNet121(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = tf.keras.applications.densenet.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)

    if num_classes == 2:
        outputs = layers.Dense(
            1,
            activation="sigmoid"
        )(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(
            num_classes,
            activation="softmax"
        )(x)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(
        inputs,
        outputs,
        name="DenseNet121_Classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss=loss,
        metrics=["accuracy"]
    )

    return model


def build_vgg16(num_classes):

    base_model = tf.keras.applications.VGG16(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = tf.keras.applications.vgg16.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)

    if num_classes == 2:
        outputs = layers.Dense(
            1,
            activation="sigmoid"
        )(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(
            num_classes,
            activation="softmax"
        )(x)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(
        inputs,
        outputs,
        name="VGG16_Classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss=loss,
        metrics=["accuracy"]
    )

    return model
