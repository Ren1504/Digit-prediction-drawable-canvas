import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from keras.models import load_model
import cv2 

st.set_page_config(layout = 'centered')

st.title("# Digits prediction using MNIST")
st.subheader("About MNIST")
st.write("""The MNIST database (Modified National Institute of Standards and Technology database) 
                is a large database of handwritten digits that is 
                commonly used for training various image processing systems. 
                The database is also widely used for training and testing in 
                the field of machine learning.""")

st.subheader("How to Use")
st.write("1.Draw a single digit in the range 0-9")
st.write("2.The digit should be drawn in the middle")
st.write("3.The digit should cover nearly 60 to 75 percent of the canvas")
st.subheader("Example")
st.image("https://media.discordapp.net/attachments/841663932204711988/1083083409011785818/image.png?width=150&height=150")
st.write("4.Lastly click on the predict button to see the prediction")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0.3)",
    stroke_width=32,
    stroke_color="white",
    background_color= "black",
    update_streamlit=True,
    height = 650,
    width = 650,
    drawing_mode="freedraw",
    display_toolbar= True
)

model = load_model("mnist.h5")

if canvas_result.image_data is not None:
    img=cv2.resize(canvas_result.image_data.astype(np.uint8),(28,28))
    img_rescalling= (cv2.resize(img, dsize=(200,200),interpolation=cv2.INTER_NEAREST))
    
    if st.button("Predict"):
      x_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      y_pred=model.predict(x_img.reshape(1,28,28))
      y_pred=np.argmax(y_pred,axis=1)
      st.header("The predicted digit is {}".format(y_pred[0]))


