import matplotlib.pyplot as plt

def display(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.legend(["Train", "Validation"])
    plt.show()
    
