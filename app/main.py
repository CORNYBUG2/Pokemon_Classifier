from dataset import preprocess
from app.model import createmodel
from app.train import trainmodel
from app.visual import display
import warnings
warnings.filterwarnings("ignore")



datasetpath = "PokemonDataset"

train_ds, val_ds, class_names = preprocess(datasetpath)

model = createmodel(len(class_names))

history = trainmodel(model, train_ds, val_ds)

display(history)

model.save("pokemon_class.keras")