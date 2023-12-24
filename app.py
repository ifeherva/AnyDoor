from typing import Tuple

import numpy as np
import streamlit as st
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from anydoorpredictor import AnydoorPredictor
from segmentation.densepose_segmenter import DenseposeSegmenter, draw_densepose
from segmentation.human_segmenter import ClothSegmenterHF, draw_semantic_mask

st.set_page_config(page_title='Icon AI Try-on demo',
                   page_icon=None,
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

with st.sidebar:
    ddim_steps = st.number_input(
        'Number of DDIM steps',
        min_value=1,
        max_value=1000,
        value=50,
    )

    guidance_scale = st.number_input(
        'Guidance scale',
        min_value=0.1,
        max_value=30.,
        value=9.,
    )

    eta = st.number_input(
        'DDIM eta',
        min_value=0.0,
        max_value=1.0,
        value=0.0,
    )

    control_strength = st.number_input(
        'Control Strength',
        min_value=0.0,
        max_value=2.0,
        value=1.0,
    )


@st.cache_resource
def load_model():
    return AnydoorPredictor()


@st.cache_resource
def load_cloth_segmenter():
    return ClothSegmenterHF()


@st.cache_resource
def load_densepose_segmenter():
    return DenseposeSegmenter()


segmenter = load_cloth_segmenter()
densepose_segmenter = load_densepose_segmenter()
anydoor_model = load_model()

person_image = None
garment_image = None
pose_person = None
pose_garment = None
mask_person = None
mask_garment = None
dp_mask_person = None
dp_mask_garment = None

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("Person image")
        person_uploaded_file = st.file_uploader(
            'Upload an image',
            type=['png', 'jpg'],
            help='Drop an image from your computer to start the process',
            key='person_uploader')
        if person_uploaded_file is not None:
            person_image = Image.open(person_uploaded_file).convert('RGB')
            st.image(person_image)

    with col2:
        st.header("Apparel image")
        garment_uploaded_file = st.file_uploader(
            'Upload an image',
            type=['png', 'jpg'],
            help='Drop an image from your computer to start the process',
            key='garment_uploader')
        if garment_uploaded_file is not None:
            garment_image = Image.open(garment_uploaded_file).convert('RGB')
            st.image(garment_image)


@st.cache_data
def segment_images(person_image: Image.Image,
                   garment_image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Generate cloth segmentation masks"""
    mask_person = segmenter(person_image)
    mask_garment = segmenter(garment_image)

    with st.container():
        st.header('Segment clothing')
        col5, col6 = st.columns(2)

        with col5:
            st.image(draw_semantic_mask(person_image, mask_person))

        with col6:
            st.image(draw_semantic_mask(garment_image, mask_garment))
    return mask_person, mask_garment


@st.cache_data
def densepose_images(person_image, garment_image):
    """Generate densepose segmentation masks"""
    dp_mask_person = densepose_segmenter(person_image)
    dp_mask_garment = densepose_segmenter(garment_image)

    with st.container():
        st.header('Dense pose detection')
        col5, col6 = st.columns(2)

        with col5:
            st.image(draw_densepose(person_image, dp_mask_person[:, :, 0]))

        with col6:
            st.image(draw_densepose(garment_image, dp_mask_garment[:, :, 0]))
    return dp_mask_person, dp_mask_garment


@st.cache_data
def run_anydoor(
        person_image,
        garment_image,
        mask_person,
        mask_garment,
        dp_mask_person,
        dp_mask_garment,
        btype,
        save_memory=False,
        guidance_scale=9,
        eta=1,
        strength=1,
        ddim_steps=50,
        num_samples=1,
) -> Image.Image:
    prediction = anydoor_model.predict(
        garment_image,
        mask_garment,
        dp_mask_garment,
        person_image,
        mask_person,
        dp_mask_person,
        btype,
        save_memory=save_memory,
        guidance_scale=guidance_scale,
        eta=eta,
        strength=strength,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
    )

    return prediction


genre = st.radio("Apparel to process", ["full_body", "upper_body"])


if st.button('Generate try-on',
             use_container_width=True,
             disabled=not (person_uploaded_file and garment_uploaded_file)):

    # Generate human mask
    with st.spinner('Running human segmentation...'):
        mask_person, mask_garment = segment_images(person_image, garment_image)

    # Generate dense pose mask
    with st.spinner('Running dense pose segmentation...'):
        dp_mask_person, dp_mask_garment = densepose_images(person_image, garment_image)

    pred_image = run_anydoor(
        person_image,
        garment_image,
        mask_person,
        mask_garment,
        dp_mask_person,
        dp_mask_garment,
        genre,
        save_memory=False,
        guidance_scale=guidance_scale,
        eta=eta,
        strength=control_strength,
        ddim_steps=ddim_steps,
        num_samples=1,
    )

    st.image(pred_image)
