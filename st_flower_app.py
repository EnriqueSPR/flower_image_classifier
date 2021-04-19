
import streamlit as st 
from PIL import Image
import tensorflow as tf 
import time

def process_image(img, img_size=(299, 299)):
    """
    This function is used to pre-process any chosen picture by the user
    to the appropriate format that the model accepts.
    Parameters:
    img: The input Image, opened using the PIL library
    img_size: Defaults to (299, 299) because it is the size that the 
                model accepts
    Returns:
    An Image array that is ready to be fed into the model.
    """
    # reshapes the image
    image = ImageOps.fit(img, img_size, Image.ANTIALIAS)
    # converts the image into numpy array
    image = np.asarray(image)
    # converts image from BGR color space to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img_resize[np.newaxis,...]
    
    return img_reshape

def top_3_predictions(array, class_list):
    inx = array[0].argsort()[-3:][::-1] # getting the indexes of the top 3 predictions in descending order
    top_1 = array[0][inx[0]]*100
    top_2 = array[0][inx[1]]*100
    top_3 = array[0][inx[2]]*100
    class_1 = class_list[inx[0]]
    class_2 = class_list[inx[1]]
    class_3 = class_list[inx[2]]
    return print("Top 1 Prediction: With {:5.2f}% probability is a picture of {}.\nTop 2 Prediction: With {:5.2f}% probability is a picture of {}.\nTop 3 Prediction: With {:5.2f}% probability is a picture of {}.".format(top_1, class_1, top_2, class_2, top_3, class_3))

def prediction_result(model, image_data):
    """
    The function that returns the prediction result from the model
    Parameters:
    model: The model to be used to classify
    image_data: Image array that is returned by the process_image function
  
    Returns --> Dictionary with class and accuracy values
    """
  # Mapping prediction results to the Flower type
 

    list_flowers = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
    pred_array = model.predict(image_data)
    
    return top_3_predictions(pred_array, list_flowers)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Flower Classifier")

st.write("This app can predict flowers from five categories: Daisy, Rose, Sunflower, Tulip and Dandelion")
st.write("Disclaimer: May not always give correct prediction!")
st.write("Made by: Rupak Karki")
st.markdown("[rupakkarki.com.np](https://www.rupakkarki.com.np)")

img = st.file_uploader("Please upload Image", type=["jpeg", "jpg", "png"])

# Display Image
st.write("Uploaded Image")
try:
	img = Image.open(img)
	st.image(img)	# display the image
	img = process_image(img)


	# Prediction
	model = tf.keras.models.load_model("my_flower_model.h5")
	prediction = prediction_result(model, img)

	# Progress Bar
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.05)
		my_bar.progress(percent_complete + 1)

	# Output
	st.write("# Flower Type: {}".format(prediction["class"]))
	st.write("With Accuracy:", prediction["accuracy"],"%")
except AttributeError:
	st.write("No Image Selected")

