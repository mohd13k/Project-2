import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Paths
test_image_paths = [

    '/Users/mohammadkhan/Desktop/AER850/Data/test/test_crack.jpg',
    '/Users/mohammadkhan/Desktop/AER850/Data/test/test_missinghead.jpg',
    '/Users/mohammadkhan/Desktop/AER850/Data/test/test_paintoff.jpg'
]

model = load_model('mohammad_model.h5')

class_labels = ['Crack', 'Missing Head', 'Paint-off']

def predict_image(model, image_path):
    img = load_img(image_path, target_size=(500, 500))  
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_labels[predicted_class]
    return predictions[0], predicted_label

# Tests model
for image_path in test_image_paths:
    print(f"Testing image: {image_path}")
    probabilities, predicted_label = predict_image(model, image_path)
    
    # Display the probabilities
    print(f"Predicted Classification Label: {predicted_label}")
    print("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {class_labels[i]}: {prob * 100:.2f}%")

    # Image Plot
    img = load_img(image_path, target_size=(500, 500))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label}\n" +
              "\n".join([f"{class_labels[i]}: {prob * 100:.1f}%" for i, prob in enumerate(probabilities)]),
              fontsize=10, color='green')
    plt.show()
