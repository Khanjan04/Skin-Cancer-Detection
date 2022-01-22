import cv2
import tensorflow as tf

CATEGORIES = ["B", "M"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("CNN.model")

prediction = model.predict([prepare('B_ISIC_0000018_downsampled.jpg')])

print(CATEGORIES[int(prediction[0][0])])
