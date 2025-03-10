# Python In-built packages
from pathlib import Path
import PIL
import google.generativeai as genai
# External packages
import streamlit as st
import os 
# Local Modules
import settings
import helper
# Initialize Google Gemini Pro
genai.configure(api_key="AIzaSyATeuavfgyi58IOrWbYnikFchU4BCoZuhw")
# Create the model with the configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

modell = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
def inputdisease(disease_detected):

    prompt = f"""
        In coconut tree {disease_detected} disease is caused, can you tell me why it is happened in the tree, can you tell me me the cause and why it is happened in 50 words.
        """
        
    # Generate the response from the model
    response = modell.generate_content(prompt)
    return response.text


# Setting page layout
st.set_page_config(
    page_title="Coconut Disease Detection",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Coconut Disease Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 10)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                names = res[0].names
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                class_detections_values = []
                for k, v in names.items():
                    class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                # create dictionary of objects detected per class
                classes_detected = dict(zip(names.values(), class_detections_values))
                # st.text(classes_detected)
                if 'bud root dropping' in classes_detected:
                    disease_detected = classes_detected['bud root dropping']
                    if disease_detected:
                        if not disease_detected == 0:
                            cause = inputdisease(disease_detected="bud root dropping")
                            st.info(f"ExplainableAI :{cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[BUD ROOT DROPPING] : {disease_detected}")
                            st.warning(f"#### [Cause]: \n Related to poor root health, inadequate nutrients, or environmental stress")
                            st.success(f"#### [Solution]: \n 1. Conduct systematic inspections of the coconut trees to spot early signs of disease or pest infestations. \n 2.  Regularly test soil to determine nutrient needs and adjust fertilization accordingly. \n 3. Implement efficient irrigation systems to avoid water stress and waterlogging. \n 4. Employ integrated pest management (IPM) strategies to control pests that can act as disease vectors.")
                            st.link_button("Go to Product Link","https://cultree.in/products/shivalik-roksin-thiophanate-methyl-70-wp-fungicide")
                if 'bud rot' in classes_detected:
                    disease_detected = classes_detected['bud rot']
                    if disease_detected:
                        if not disease_detected == 0:
                            cause = inputdisease(disease_detected="bud rot")
                            st.info(f"ExplainableAI :{cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[BUD ROT] : {disease_detected}")
                            st.warning(f"#### [Cause] : \n Bud rot is caused by the fungus Phytophthora palmivora. It leads to black lesions on young fronds and leaves, weakening the tree.")
                            st.success(f"#### [Soultion] : \n 1. Regularly inspect your coconut trees for signs of black lesions \n 2. Avoid water stress and waterlogging. \n 3. Some of the more common coconut tree disease issues include fungal or bacterial problems")
                            st.link_button("Go to Product Link","https://cultree.in/products/shivalik-roksin-thiophanate-methyl-70-wp-fungicide")
                if 'gray leaf spot' in classes_detected:
                    disease_detected = classes_detected['gray leaf spot']
                    if disease_detected:
                        if not disease_detected == 0:
                            cause = inputdisease(disease_detected="gray leaf spot")
                            st.info(f"ExplainableAI :{cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[GRAY LEAF SPOT] : {disease_detected}")
                            st.warning(f"#### [Cause] \n Gray leaf spots are caused by both fungi and bacteria. Circular or elongated spots develop on foliage.")
                            st.success(f"#### [Soultion] \n 1. Pestalotiopsis palmarum is the primary causative agent of gray leaf spot. It affects not only coconut trees but also bananas and date palms. The fungus leads to leaf spots, petiole/rachis blights, and sometimes bud rot in palms. \n 2. Gray leaf spot tends to be more severe on older leaves. Unfavorable growing conditions, such as excessive moisture or poor ventilation, can exacerbate the disease. \n 3. Rain and wind play a crucial role in spreading the spores of both brown leaf spot and gray leaf spot fungi. Consequently, these diseases are more common during wet weather.")
                            st.link_button("Go to Product Link","https://www.bighaat.com/products/agriventure-cooxy")
                if 'leaf rot' in classes_detected:
                    disease_detected = classes_detected['leaf rot']
                    if disease_detected:
                        if not disease_detected == 0:
                            cause = inputdisease(disease_detected="leaf rot")
                            st.info(f"ExplainableAI :{cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[LEAF ROT] : {disease_detected}")
                            st.warning(f"#### [Cause] \n This type of disease is majorly caused by fungi, Peastalozzia palmarum, and the Bipolaris incurvata. Initially, there will be a visible appearance of the yellow-brown spots that appear on top of the leaflets from the lower fronds. This then gradually enlarges and tends to turn grey on the coconut planting.")
                            st.success(f"#### [Soultion] \n 1. Inspect your coconut tree during wet weather, as rain and wind can spread spores that cause leaf spots. \n 2. Apply fungicides to control leaf spot diseases \n 3. If your coconut tree exhibits a condition known as “pencil point disorder,” it may be due to a lack of micronutrients.")
                            st.link_button("Go to Product Link","https://ariesagro.com/jahaan-hexaconazole5-w-w/")
                if 'stembleeding' in classes_detected:
                    disease_detected = classes_detected['stembleeding']
                    if disease_detected:
                        if not disease_detected == 0:
                            cause = inputdisease(disease_detected="stembleeding")
                            st.info(f"ExplainableAI :{cause}")
                            st.title(f"#### Disease Detected: ##### \n :blue[STEMBLEEDING] : {disease_detected}")
                            st.warning(f"#### [Cause] \n Stem bleeding disease of coconut, caused by Thielaviopsis paradoxa (de Seyness) Von Hohnel, is widely prevalent in all coconut growing regions in the tropics.")
                            st.success(f"#### [Soultion] \n 1. Chisel out the affected tissues completely. \n 2. Paint the wound with Bordeaux paste or apply coal tar after 1-2 days. \n 3. Apply neem cake (approximately 5 kg per palm) to the basin along with other organic materials. \n 4. Root feed with Tridemorph (Calixin-5%) in water (5 ml in 100 ml) thrice a year during April-May, September-October, and January-February.")
                            st.link_button("Go to Product 1 Link","https://www.bighaat.com/products/blue-copper-fungicide-1")
                            st.link_button("Go to Product 2 Link","http://www.rayfull.com/Productshows.asp?ID=338")
                            st.link_button("Go to Product 3 Link","https://krishisevakendra.in/products/bordeaux-mixture")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

