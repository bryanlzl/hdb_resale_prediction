import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import joblib

# Load the trained models
resale_linear_model = joblib.load("resale_linear_model.joblib")
rental_random_forest_model = joblib.load("rental_random_forest_model.joblib")


# Define all features
all_features = [
    "year",
    "storey_range",
    "floor_area_sqm",
    "remaining_lease",
    "mrt_dist",
    "shopping_dist",
    "school_dist",
    "hawker_dist",
]
# Dropdown options (town and flat type)
town_options = {
    "ANG_MO_KIO": "Ang Mo Kio",
    "BEDOK": "Bedok",
    "BISHAN": "Bishan",
    "BUKIT_BATOK": "Bukit Batok",
    "BUKIT_MERAH": "Bukit Merah",
    "BUKIT_PANJANG": "Bukit Panjang",
    "BUKIT_TIMAH": "Bukit Timah",
    "CENTRAL_AREA": "Central Area",
    "CHOA_CHU_KANG": "Choa Chu Kang",
    "CLEMENTI": "Clementi",
    "GEYLANG": "Geylang",
    "HOUGANG": "Hougang",
    "JURONG_EAST": "Jurong East",
    "JURONG_WEST": "Jurong West",
    "KALLANG_WHAMPOA": "Kallang / Whampoa",
    "MARINE_PARADE": "Marine Parade",
    "PASIR_RIS": "Pasir Ris",
    "PUNGGOL": "Punggol",
    "QUEENSTOWN": "Queenstown",
    "SEMBAWANG": "Sembawang",
    "SENGKANG": "Sengkang",
    "SERANGOON": "Serangoon",
    "TAMPINES": "Tampines",
    "TOA_PAYOH": "Toa Payoh",
    "WOODLANDS": "Woodlands",
    "YISHUN": "Yishun",
}
flat_type_options = {
    "1_ROOM": "1 Room",
    "2_ROOM": "2 Room",
    "3_ROOM": "3 Room",
    "4_ROOM": "4 Room",
    "5_ROOM": "5 Room",
    "EXECUTIVE": "Executive",
    "MULTI_GENERATION": "Multi-Generation",
}
feature_names = {
    "year": "Sale Year",
    "town": "Town",
    "flat_type": "Flat Type",
    "storey_range": "Storey Range",
    "floor_area_sqm": "Floor Area (m^2)",
    "remaining_lease": "Remaining Lease (yr)",
    "mrt_dist": "Distance from MRT (km)",
    "shopping_dist": "Distance from Shopping Amenities (km)",
    "school_dist": "Distance from School (km)",
    "hawker_dist": "Distance from Hawker Centre (km)",
}
feature_units = {
    "year": "year",
    "town": "-",
    "flat_type": "-",
    "storey_range": "-",
    "floor_area_sqm": "m^2",
    "remaining_lease": "year",
    "mrt_dist": "km",
    "shopping_dist": "km",
    "school_dist": "km",
    "hawker_dist": "km",
}



rental_all_features = [
    "property_age",
    "avg_floor_area_sqm",
    "mrt_dist",
    "shopping_dist",
    "intschool_dist",
    "hawker_dist",
    "months_since_signedrental",
]
rental_town_options = {
    "ANG_MO_KIO": "Ang Mo Kio",
    "BEDOK": "Bedok",
    "BISHAN": "Bishan",
    "BUKIT_BATOK": "Bukit Batok",
    "BUKIT_MERAH": "Bukit Merah",
    "BUKIT_PANJANG": "Bukit Panjang",
    "BUKIT_TIMAH": "Bukit Timah",
    "CENTRAL": "Central Area",
    "CHOA_CHU_KANG": "Choa Chu Kang",
    "CLEMENTI": "Clementi",
    "GEYLANG": "Geylang",
    "HOUGANG": "Hougang",
    "JURONG_EAST": "Jurong East",
    "JURONG_WEST": "Jurong West",
    "KALLANG_WHAMPOA": "Kallang / Whampoa",
    "MARINE_PARADE": "Marine Parade",
    "PASIR_RIS": "Pasir Ris",
    "PUNGGOL": "Punggol",
    "QUEENSTOWN": "Queenstown",
    "SEMBAWANG": "Sembawang",
    "SENGKANG": "Sengkang",
    "SERANGOON": "Serangoon",
    "TAMPINES": "Tampines",
    "TOA_PAYOH": "Toa Payoh",
    "WOODLANDS": "Woodlands",
    "YISHUN": "Yishun",
}
rental_flat_type_options = {
    "1_ROOM": "1 Room",
    "2_ROOM": "2 Room",
    "3_ROOM": "3 Room",
    "4_ROOM": "4 Room",
    "5_ROOM": "5 Room",
    "EXECUTIVE": "Executive",
}
rental_feature_names = {
    "town": "Town",
    "flat_type": "Flat Type",
    "storey_range": "Storey Range",
    "avg_floor_area_sqm": "Floor Area (m^2)",
    "remaining_lease": "Remaining Lease (yr)",
    "mrt_dist": "Distance from MRT (km)",
    "shopping_dist": "Distance from Shopping Amenities (km)",
    "intschool_dist": "Distance from International School (km)",
    "hawker_dist": "Distance from Hawker Centre (km)",
    "months_since_signedrental": "Rental Start Date",
    "property_age": "Property Age (yr)"
}
rental_feature_units = {
    "town": "-",
    "flat_type": "-",
    "storey_range": "-",
    "avg_floor_area_sqm": "m^2",
    "remaining_lease": "year",
    "mrt_dist": "km",
    "shopping_dist": "km",
    "intschool_dist": "km",
    "hawker_dist": "km",
    "property_age": "year",
    "months_duration": "month"
}


def input_field(field_name, user_inputs):
    if prediction_mode == "Resale Price":
        unique_key = f"input_{field_name}"
        if field_name == "town":
            selected_town = user_inputs.get("town", list(town_options.keys())[0])
            return st.sidebar.selectbox(
                "Town",
                list(town_options.values()),
                index=list(town_options.keys()).index(selected_town),
                key=unique_key,
            )
        elif field_name == "flat_type":
            selected_flat_type = user_inputs.get(
                "flat_type", list(flat_type_options.keys())[0]
            )
            return st.sidebar.selectbox(
                "Flat Type",
                list(flat_type_options.values()),
                index=list(flat_type_options.keys()).index("3_ROOM"),
                key=unique_key,
            )
        elif field_name == "storey_range":
            return st.sidebar.number_input(
                feature_names[field_name],
                min_value=1,
                max_value=50,
                key=field_name,
            )
        elif field_name == "remaining_lease":
            return st.sidebar.number_input(
                feature_names[field_name],
                min_value=0,
                max_value=99,
                value=90,
                key=field_name,
            )
        elif field_name == "year":
            return (
                st.sidebar.number_input(
                    feature_names[field_name],
                    min_value=int(min_value(field_name)),
                    max_value=int(max_value(field_name)),
                    value=2023,
                    key=field_name,
                )
                - 2024
            )
        elif field_name.endswith("_dist"):
            return st.sidebar.number_input(
                feature_names[field_name],
                min_value=0.0,
                value=1.0,
                step=0.001,
                format="%.2f",
                key=field_name,
            )
        elif field_name == "floor_area_sqm":
            return st.sidebar.number_input(
                feature_names[field_name],
                min_value=0.0,
                value=65.0,
                step=0.1,
                key=field_name,
            )
        else:
            raise ValueError(f"Unknown field type for {field_name}")
        
    elif prediction_mode == "Rental Price":
        unique_key = f"input_{field_name}"
        if field_name == "town":
            selected_town = user_inputs.get("town", list(rental_town_options.keys())[0])
            return st.sidebar.selectbox(
                "Town",
                list(rental_town_options.values()),
                index=list(rental_town_options.keys()).index(selected_town),
                key=unique_key,
            )
        elif field_name == "flat_type":
            selected_flat_type = user_inputs.get(
                "flat_type", list(rental_flat_type_options.keys())[0]
            )
            return st.sidebar.selectbox(
                "Flat Type",
                list(rental_flat_type_options.values()),
                index=list(rental_flat_type_options.keys()).index("3_ROOM"),
                key=unique_key,
            )
        elif field_name in ["storey_range", "remaining_lease"]:
            return st.sidebar.number_input(
                rental_feature_names[field_name],
                min_value=int(min_value(field_name)),
                max_value=int(max_value(field_name)),
                key=field_name,
            )
        elif field_name == "year":
            return (
                st.sidebar.number_input(
                    rental_feature_names[field_name],
                    min_value=int(min_value(field_name)),
                    max_value=int(max_value(field_name)),
                    key=field_name,
                )
                - 2024
            )
        elif field_name.endswith("_dist"):
            return st.sidebar.number_input(
                rental_feature_names[field_name],
                min_value=0.0,
                value=1.0,
                step=0.001,
                format="%.2f",
                key=field_name,
            )
        elif field_name == "avg_floor_area_sqm":
            return st.sidebar.number_input(
                rental_feature_names[field_name],
                min_value=0.0,
                value=65.0,
                step=1.0,
                key=field_name,
            )
        elif field_name == "property_age":
            return st.sidebar.number_input(
                rental_feature_names[field_name],
                min_value=0,
                value=90,
                step=1,
                key=field_name,
            )
        elif field_name == "months_since_signedrental":
            return st.sidebar.date_input(
            "Select Rental Start Date",
            min_value=date(2010, 1, 1),
            max_value=date(2029, 12, 31),
            value=date(2023, 8, 9),
            format='DD/MM/YYYY',
            help="Select the month and year when the rental agreement was signed."
        )
        else:
            raise ValueError(f"Unknown field type for {field_name}")


def render_sidebar():
    st.sidebar.title("HDB Price Prediction Application")
    prediction_mode = st.sidebar.radio(
        "mode select",
        ("Resale Price", "Rental Price"),
        horizontal=True,
        label_visibility="hidden",
    )
    st.sidebar.text("\n")
    st.sidebar.title("Input HDB parameters")
    return prediction_mode


def render_sidebar2():
    if prediction_mode == "Resale Price":
        st.sidebar.markdown("* HDB parameters for resale price prediction")
        st.sidebar.markdown("* Ensure all fields are filled before predicting")
        # Fetch town and flat_type selection
        selected_town = input_field("town", {})
        selected_flat_type = input_field("flat_type", {})

        # Resolve actual keys from options
        town_key = next(
            key for key, value in town_options.items() if value == selected_town
        )
        flat_type_key = next(
            key
            for key, value in flat_type_options.items()
            if value == selected_flat_type
        )

        # Initialize user_inputs with town and flat_type selections
        user_inputs = {"town": town_key, "flat_type": flat_type_key}

        for feature in all_features:
            if feature not in [
                "town",
                "flat_type",
            ]:  # Avoid re-fetching town and flat_type
                user_inputs[feature] = input_field(feature, user_inputs)
        return user_inputs

    elif prediction_mode == "Rental Price":
        st.sidebar.markdown("* HDB parameters for rental price prediction")
        st.sidebar.markdown("* Ensure all fields are filled before predicting")
        
        selected_town = input_field("town", {})
        selected_flat_type = input_field("flat_type", {})
        town_key = next(
            key for key, value in rental_town_options.items() if value == selected_town
        )
        flat_type_key = next(
            key
            for key, value in rental_flat_type_options.items()
            if value == selected_flat_type
        )

        user_inputs = {"town": town_key, "flat_type": flat_type_key}

        for feature in rental_all_features:
            if feature == "months_since_signedrental":
                #reference_date = date(2023, 12, 31)
                input_date = input_field(feature, user_inputs)
                user_inputs[feature] = (input_date.year - 2023) * 12 + (input_date.month - 12)
            elif feature not in [
                "town",
                "flat_type",
            ]:  # Avoid re-fetching town and flat_type'
                user_inputs[feature] = input_field(feature, user_inputs)

        return user_inputs


def min_value(field_name):
    if field_name == "year":
        return 2010
    elif field_name == "storey_range":
        return 1
    elif field_name == "remaining_lease":
        return 0


def max_value(field_name):
    if field_name == "year":
        return 2030
    elif field_name == "storey_range":
        return 50
    elif field_name == "remaining_lease":
        return 99


def predict_price(user_inputs):
    if prediction_mode == "Resale Price":
        town = user_inputs.pop("town")
        flat_type = user_inputs.pop("flat_type")

        # One-hot encode the categorical variables
        for town_option in town_options:
            if town_option != "ANG_MO_KIO":
                user_inputs[f"town_{town_option}"] = 1 if town_option == town else 0
        for flat_type_option in flat_type_options:
            if flat_type_option != "1_ROOM":
                user_inputs[f"flat_type_{flat_type_option}"] = (
                    1 if flat_type_option == flat_type else 0
                )
        data_for_prediction = pd.DataFrame([user_inputs])
        # Predict the log resale price
        predicted_log_price = resale_linear_model.predict(data_for_prediction)
        # Convert predicted log price -> actual resale price
        predicted_price = np.exp(predicted_log_price)
        return predicted_price
    elif prediction_mode == "Rental Price":
        town = user_inputs.pop("town")
        flat_type = user_inputs.pop("flat_type")

        # One-hot encode the categorical variables
        for town_option in rental_town_options:
            if town_option != "ANG_MO_KIO":
                user_inputs[f"town_{town_option}"] = 1 if town_option == town else 0
        for flat_type_option in rental_flat_type_options:
            if flat_type_option != "1_ROOM":
                user_inputs[f"flat_type_{flat_type_option}"] = (
                    1 if flat_type_option == flat_type else 0
                )
        data_for_prediction = pd.DataFrame([user_inputs])
        data_for_prediction.to_csv("TESTEST.csv", index=False)
        # Predict the log rental price
        predicted_price = rental_random_forest_model.predict(data_for_prediction)
        # Convert predicted log price -> actual rental price
        return predicted_price


def display_inputs_table(user_inputs):
    if prediction_mode == "Resale Price":
        inputs_df = pd.DataFrame(list(user_inputs.items()), columns=["Feature", "Value"])

        year_index = inputs_df[inputs_df["Feature"] == "year"].index
        if not year_index.empty:
            inputs_df.loc[year_index, "Value"] = inputs_df.loc[year_index, "Value"] + 2024

        FT_index = inputs_df[inputs_df["Feature"] == "flat_type"].index
        if not FT_index.empty:
            inputs_df.loc[FT_index, "Value"] = inputs_df.loc[FT_index, "Value"].apply(
                lambda x: x.replace("_", " ").lower()
            )

        town_index = inputs_df[inputs_df["Feature"] == "town"].index
        if not year_index.empty:
            inputs_df.loc[town_index, "Value"] = inputs_df.loc[town_index, "Value"].apply(
                lambda x: x.replace("_", " ").title()
            )

        inputs_df["Unit"] = inputs_df["Feature"].apply(lambda x: feature_units.get(x, "-"))
        inputs_df["Feature"] = inputs_df["Feature"].apply(
            lambda x: feature_names.get(x, "-")
        )
        st.write("### Resale price prediction parameters:")
        st.table(inputs_df)
    elif prediction_mode == "Rental Price":
        inputs_df = pd.DataFrame(list(user_inputs.items()), columns=["Feature", "Value"])
        year_index = inputs_df[inputs_df["Feature"] == "year"].index
        if not year_index.empty:
            inputs_df.loc[year_index, "Value"] = inputs_df.loc[year_index, "Value"] + 2024

        FT_index = inputs_df[inputs_df["Feature"] == "flat_type"].index
        if not FT_index.empty:
            inputs_df.loc[FT_index, "Value"] = inputs_df.loc[FT_index, "Value"].apply(
                lambda x: x.replace("_", " ").lower()
            )

        town_index = inputs_df[inputs_df["Feature"] == "town"].index
        if not town_index.empty:
            inputs_df.loc[town_index, "Value"] = inputs_df.loc[town_index, "Value"].apply(
                lambda x: x.replace("_", " ").title()
            )

        # Convert months_since_signedrental to date string
        msr_index = inputs_df[inputs_df["Feature"] == "months_since_signedrental"].index
        if not msr_index.empty:
            inputs_df.loc[msr_index, "Value"] = inputs_df.loc[msr_index, "Value"].apply(convert_months_to_date)

        inputs_df["Unit"] = inputs_df["Feature"].apply(lambda x: rental_feature_units.get(x, "-"))
        inputs_df["Feature"] = inputs_df["Feature"].apply(
            lambda x: rental_feature_names.get(x, "-")
        )
        st.write("### Rental price prediction parameters:")
        st.table(inputs_df)


def convert_months_to_date(months_since):
    # Reference date is December 31, 2023
    reference_date = date(2023, 12, 31)
    # Calculate the actual date by adding the months
    target_date = reference_date + relativedelta(months=months_since)
    # Format the date into "Month Year"
    return target_date.strftime("%b %Y")


def convert_duration_to_date(months_duration):
    reference_date = date(2023, 12, 31)
    target_date = reference_date + relativedelta(months=months_duration)
    month_year_string = target_date.strftime('%B %Y')
    return month_year_string


def prediction_results(user_inputs):
    if prediction_mode == "Resale Price":
        predicted_price = predict_price(user_inputs)
        if all(
            value == 0 for key, value in user_inputs.items() if key.startswith("flat_type_")
        ):
            flat_type = "1 room"
        else:
            flat_type = (
                [
                    key[9:]
                    for key, value in user_inputs.items()
                    if key.startswith("flat_type") and value == 1
                ][0]
                .replace("_", " ")
                .lower()
            )
        if all(value == 0 for key, value in user_inputs.items() if key.startswith("town_")):
            town = "Ang Mo Kio"
        else:
            town = (
                [
                    key[5:]
                    for key, value in user_inputs.items()
                    if key.startswith("town_") and value == 1
                ][0]
                .replace("_", " ")
                .title()
            )

        sale_year = user_inputs["year"] + 2024
        predicted_price_badge = f"<span style='background-color: orange; color: black; padding: 0.2em 0.5em; border-radius: 0.3em; font-weight: bold;'>${predicted_price[0]:,.2f}</span>"

        st.write(
            f"The predicted resale price of a <span style='color: orange'>{town}</span> <span style='color: orange'>{flat_type}</span> flat in <span style='color: orange'>{sale_year}</span> is <span style='color: orange'>{predicted_price_badge}</span>",
            unsafe_allow_html=True,
        )
    elif prediction_mode == "Rental Price":
        predicted_price = predict_price(user_inputs)
        if all(
            value == 0 for key, value in user_inputs.items() if key.startswith("flat_type_")
        ):
            flat_type = "1 room"
        else:
            flat_type = (
                [
                    key[9:]
                    for key, value in user_inputs.items()
                    if key.startswith("flat_type") and value == 1
                ][0]
                .replace("_", " ")
                .lower()
            )
        if all(value == 0 for key, value in user_inputs.items() if key.startswith("town_")):
            town = "Ang Mo Kio"
        else:
            town = (
                [
                    key[5:]
                    for key, value in user_inputs.items()
                    if key.startswith("town_") and value == 1
                ][0]
                .replace("_", " ")
                .title()
            )
        rental_duration = user_inputs["months_since_signedrental"]
        age = user_inputs["property_age"]
        predicted_price_badge = f"<span style='background-color: orange; color: black; padding: 0.2em 0.5em; border-radius: 0.3em; font-weight: bold;'>${predicted_price[0]:,.2f} per month</span>"

        st.write(
            f"The predicted rental price of a <span style='color: orange'>{town}</span> <span style='color: orange'>{flat_type}</span> flat with an age of <span style='color: orange'>{age} years</span> and a rental starting in <span style='color: orange'>{convert_duration_to_date(rental_duration)}</span> is <span style='color: orange'>{predicted_price_badge}</span>",
            unsafe_allow_html=True,
        )


# Main App Flow
prediction_mode = render_sidebar()
user_inputs = render_sidebar2()

if prediction_mode == "Resale Price":
    # Display the input parameters table in the main page
    display_inputs_table(user_inputs)
    st.write("### Predict with fitted linear model")
    predict_clicked = st.button("Predict Resale Price", key="predict_button")
    if predict_clicked:
        prediction_results(user_inputs)
else:
    # Display the input parameters table in the main page
    display_inputs_table(user_inputs)
    st.write("### Predict with fitted random forest model")
    predict_clicked = st.button("Predict Rental Price", key="predict_button")
    if predict_clicked:
        prediction_results(user_inputs)
