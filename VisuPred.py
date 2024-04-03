import streamlit as st;


with st.sidebar:
    st.text_input(label="Input Town", value="", key="input_town",
                type="default", autocomplete=None, 
                on_change=None, args=None, kwargs=None,
                placeholder=None, disabled=False, 
                label_visibility="visible")
    st.text_input(label="Input Flat Type", value="", key="input_flat_type",
                type="default",  autocomplete=None, 
                on_change=None, args=None, kwargs=None,
                placeholder=None, disabled=False, 
                label_visibility="visible")
    st.text_input(label="Input Block", value="", key="input_block",
                type="default", autocomplete=None, 
                on_change=None,  args=None, kwargs=None,
                placeholder=None, disabled=False, 
                label_visibility="visible")
    st.text_input(label="Input Flat Storey", value="", key="input_storey",
                type="default", autocomplete=None, 
                on_change=None,  args=None, kwargs=None,
                placeholder=None, disabled=False, 
                label_visibility="visible")
    st.text_input(label="Input Flat Storey", value="", key="input_storey",
                type="default", autocomplete=None, 
                on_change=None,  args=None, kwargs=None,
                placeholder=None, disabled=False, 
                label_visibility="visible")