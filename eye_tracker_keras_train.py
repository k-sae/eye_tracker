from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from core import *
from model import *

epochs = 10
batch_size = 20

images, predictions = load_training_dataframe()

checkpoint = ModelCheckpoint(saved_weights_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             )
csv_logger = CSVLogger('v.csv')
model = get_model()
early_stop = EarlyStopping(patience=4, monitor='val_loss')
history = model.fit(images, predictions,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    shuffle=True,
                    validation_split=0.4,
                    callbacks=[checkpoint,
                               csv_logger,
                               early_stop])

model.save_weights(saved_weights_name)
# score = model.evaluate(X_test, Y_test, verbose=0)
