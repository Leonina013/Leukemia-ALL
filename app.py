import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np

from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

def local_css(file_name):
 with open(file_name) as f:
  st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

animation_symbol = "❄"

st.markdown(
 f"""
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 <div class="snowflake">{animation_symbol}</div>
 """,
 unsafe_allow_html=True
)




def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://static.vecteezy.com/system/resources/previews/007/934/086/non_2x/abstract-purple-and-pink-gradient-waves-background-glowing-lines-on-purple-background-vector.jpg");
             background-attachment: fixed;
             
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


st.sidebar.title("ALL Detector")


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
@st.cache(allow_output_mutation=True)
def load_models():
 model = load_model('keras_Model.h5', compile=False)
 model.predict()
 model.summary()
 return model

# Load the labels
class_names = open('labels.txt', 'r').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.header('Acute Lymphoblastic Leukemia Detector')
st.subheader('Make sure that you put in a picture of a cell slide. Anything else might give a false result of a Benign tumor. You have been warned')

mode=st.sidebar.select_slider('Choose between using a jpg file or taking a picture using your camera', options=['JPG','CAMERA'])

if mode == 'CAMERA':
 uploaded_file = st.camera_input("Take a photo of the slide")


if mode == 'JPG':
 uploaded_file = st.file_uploader("Choose an Image", type="jpg")

image = Image.open(uploaded_file).convert('RGB')


#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

#turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
@st.cache
def predicts():
 prediction = model.predict(data)
 index = np.argmax(prediction)
 class_name = class_names[index]
 confidence_score = prediction[0][index]
 confidence = confidence_score
 return prediction, index, class_name, confidence_score, confidence
 print('Class:', class_name, end='')
 print('Confidence score:', confidence_score)



st.title("ALL Classification")
st.header("Benign ALL Vs Malignant ALL")
st.text("Upload the picture of a slide to detect it is normal or has ALL")
# file upload and handling logic


if uploaded_file is not None:

 image = Image.open(uploaded_file).convert('RGB')
#image = Image.open(img_name).convert('RGB')
 st.image(image, caption='Uploaded an image.', use_column_width=True)
 st.write("")
 st.write("Doing the classification......Calling the pathologist with his cup of SwissMiss")

 sex = st.sidebar.selectbox('What is the sex of the person?',('Male', 'Female'))
 age = st.sidebar.slider("What is the age of the person?",0.0,150.0,50.0)
 smoker = st.sidebar.selectbox('Is the Person a Smoker',('Yes', 'No'))

@st.cache
def additional_stats():
 if age <= 5.0 and sex == 'Male':
  confidence = confidence_score * ((165+age)/100)
 if age > 5.0 and age <= 25.0 and sex == 'Male':
  confidence = confidence_score * ((100-age)/100)
 if age > 25.0 and age <=50.0 and sex == 'Male':
  confidence = confidence_score * ((60+age)/100)
 if age > 50.0 and sex == 'Male':
  confidence = confidence_score * ((70+age)/100)
 if age <= 5.0 and sex == 'Female':
  confidence = confidence_score * ((160+age)/100)
 if age > 5.0 and age <= 50.0 and sex == 'Female':
  confidence = confidence_score * ((95-age)/100)
 if age > 25.0 and age <=50.0 and sex == 'Female':
  confidence = confidence_score * ((50+age)/100)
 if age > 50.0 and sex == 'Female':
  confidence = confidence_score * ((65+age)/100)

 if smoker == 'Yes' and age >= 60:
  confidence2 = confidence*3.4
  st.write("Smoking increases your chance of having ALL 3.4 times your actual probability after all other factors like age and gender have been considered")
 if smoker == 'Yes' and age < 60:
  confidence2 = confidence*1.72
  st.write("Smoking increases your chance of having ALL 1.72 times your actual probability after all other factors like age and gender have been considered")
 if smoker == 'No':
  confidence2 = confidence
 #https://pubmed.ncbi.nlm.nih.gov/8246285/#:~:text=However%2C%20among%20participants%20aged%2060,CI%20%3D%200.97%2D11.9).
 return confidence2


if confidence_score >= 0.85:

 if index == 0:
  st.write('You identify as', sex,'and your age is', age,'years')
  st.write('The Predicted Class is:', class_name)
  st.write('Probability Percentage:', confidence_score*100, '%')
  st.write("You may have a benign condition, or are free of ALL. Get it checked once during your regular body check")
  st.write("Benign tumors are those that stay in their primary location without invading other sites of the body. They do not spread to local structures or to distant parts of the body. Benign tumors tend to grow slowly and have distinct borders. Benign tumors are not usually problematic.")


 else:
  st.write('You identify as', sex,'and your age is', age,'years')
  st.write('The Predicted Class is:', class_name)
  st.write('The Original Probability Percentage:', confidence_score*100, '%')
  st.write('Probability Percentage due to your age group:', confidence2 * 100, '%')

 if index == 1:
  st.write("This looks like a Malignant Pro-B variant of ALL. You need to get it checked before the cancer starts spreading")
 elif index == 2:
  st.write("This looks like a Malignant Pre-B variant of ALL. You need to get it checked ASAP before the condition metastisizes")
 elif index == 3:
  st.write("This looks like an early Malignant Pre-B variant of ALL. You need to get it checked as a priority before it becomes something serious")
  st.write("Malignancy is a term for diseases in which abnormal cells divide without control and can invade nearby tissues. Malignant cells can also spread  to other parts of the body through the blood and lymph systems.")


else:
 st.write("You are free from ALL but don't forget to get a body checkup regularly")   






 
















