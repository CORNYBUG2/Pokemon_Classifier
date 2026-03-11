import tensorflow as tf

def preprocess(path):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "PokemonDataset",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224,224),
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "PokemonDataset",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224,224),
        batch_size=32
    )

    return train_ds, val_ds, train_ds.class_names

