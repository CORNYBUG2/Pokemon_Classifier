def trainmodel(model, train_ds, val_ds):

    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=100
    )

    return history