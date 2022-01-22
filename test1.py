from keras.models import load_model

model = load_model('train_isic.py')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/B_ISIC_0000018_downsampled.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'Malignant'
    print(prediction)
else:
    prediction = 'Benign'
    print(prediction)
