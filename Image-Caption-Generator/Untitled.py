from keras.models import load_model
import joblib

model = load_model('model.hdf5')
model.summary()
