import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv')

df=df.drop('ID_LAT_LON_YEAR_WEEK',axis=1)

kolon_adları = df.columns.tolist()
lr = LinearRegression()
imp = IterativeImputer(estimator=lr)
df_imp = imp.fit_transform(df)
df = pd.DataFrame(df_imp)
df.columns = kolon_adları
df[['year','week_no']]=df[['year','week_no']].astype(int)

x = df.drop('emission', axis=1)
y = df[['emission']]

x = x.select_dtypes(include=[np.number])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

preprocessor = StandardScaler()

# Tahmin fonksiyonu
def time_pred(latitude,longitude,year,week_no,SulphurDioxide_SO2_column_number_density,SulphurDioxide_SO2_column_number_density_amf,
              SulphurDioxide_SO2_slant_column_number_density,SulphurDioxide_cloud_fraction,SulphurDioxide_sensor_azimuth_angle,
              SulphurDioxide_sensor_zenith_angle,SulphurDioxide_solar_azimuth_angle,SulphurDioxide_solar_zenith_angle,
              SulphurDioxide_SO2_column_number_density_15km,CarbonMonoxide_CO_column_number_density,CarbonMonoxide_H2O_column_number_density,
              CarbonMonoxide_cloud_height,CarbonMonoxide_sensor_altitude,CarbonMonoxide_sensor_azimuth_angle,CarbonMonoxide_sensor_zenith_angle,
              CarbonMonoxide_solar_azimuth_angle,CarbonMonoxide_solar_zenith_angle,NitrogenDioxide_NO2_column_number_density,
              NitrogenDioxide_tropospheric_NO2_column_number_density,NitrogenDioxide_stratospheric_NO2_column_number_density,
              NitrogenDioxide_NO2_slant_column_number_density,NitrogenDioxide_tropopause_pressure,NitrogenDioxide_absorbing_aerosol_index,
              NitrogenDioxide_cloud_fraction,NitrogenDioxide_sensor_altitude,NitrogenDioxide_sensor_azimuth_angle,
              NitrogenDioxide_sensor_zenith_angle,NitrogenDioxide_solar_azimuth_angle,NitrogenDioxide_solar_zenith_angle,
              Formaldehyde_tropospheric_HCHO_column_number_density,Formaldehyde_tropospheric_HCHO_column_number_density_amf,
              Formaldehyde_HCHO_slant_column_number_density,Formaldehyde_cloud_fraction,Formaldehyde_solar_zenith_angle,
              Formaldehyde_solar_azimuth_angle,Formaldehyde_sensor_zenith_angle,Formaldehyde_sensor_azimuth_angle,UvAerosolIndex_absorbing_aerosol_index,
              UvAerosolIndex_sensor_altitude,UvAerosolIndex_sensor_azimuth_angle,UvAerosolIndex_sensor_zenith_angle,UvAerosolIndex_solar_azimuth_angle,
              UvAerosolIndex_solar_zenith_angle,Ozone_O3_column_number_density,Ozone_O3_column_number_density_amf,Ozone_O3_slant_column_number_density,
              Ozone_O3_effective_temperature,Ozone_cloud_fraction,Ozone_sensor_azimuth_angle,Ozone_sensor_zenith_angle,Ozone_solar_azimuth_angle,
              Ozone_solar_zenith_angle,UvAerosolLayerHeight_aerosol_height,UvAerosolLayerHeight_aerosol_pressure,UvAerosolLayerHeight_aerosol_optical_depth,
              UvAerosolLayerHeight_sensor_zenith_angle,UvAerosolLayerHeight_sensor_azimuth_angle,UvAerosolLayerHeight_solar_azimuth_angle,
              UvAerosolLayerHeight_solar_zenith_angle,Cloud_cloud_fraction,Cloud_cloud_top_pressure,Cloud_cloud_top_height,Cloud_cloud_base_pressure,
              Cloud_cloud_base_height,Cloud_cloud_optical_depth,Cloud_surface_albedo,Cloud_sensor_azimuth_angle,Cloud_sensor_zenith_angle,
              Cloud_solar_azimuth_angle,Cloud_solar_zenith_angle):
    input_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'year': [year],
        'week_no': [week_no],
        'SulphurDioxide_SO2_column_number_density': [SulphurDioxide_SO2_column_number_density],
        'SulphurDioxide_SO2_column_number_density_amf': [SulphurDioxide_SO2_column_number_density_amf],
        'SulphurDioxide_SO2_slant_column_number_density': [SulphurDioxide_SO2_slant_column_number_density],
        'SulphurDioxide_cloud_fraction': [SulphurDioxide_cloud_fraction],
        'SulphurDioxide_sensor_azimuth_angle': [SulphurDioxide_sensor_azimuth_angle],
        'SulphurDioxide_sensor_zenith_angle': [SulphurDioxide_sensor_zenith_angle],
        'SulphurDioxide_solar_azimuth_angle': [SulphurDioxide_solar_azimuth_angle],
        'SulphurDioxide_solar_zenith_angle': [SulphurDioxide_solar_zenith_angle],
        'SulphurDioxide_SO2_column_number_density_15km': [SulphurDioxide_SO2_column_number_density_15km],
        'CarbonMonoxide_CO_column_number_density': [CarbonMonoxide_CO_column_number_density],
        'CarbonMonoxide_H2O_column_number_density': [CarbonMonoxide_H2O_column_number_density],
        'CarbonMonoxide_cloud_height': [CarbonMonoxide_cloud_height],
        'CarbonMonoxide_sensor_altitude': [CarbonMonoxide_sensor_altitude],
        'CarbonMonoxide_sensor_azimuth_angle': [CarbonMonoxide_sensor_azimuth_angle],
        'CarbonMonoxide_sensor_zenith_angle': [CarbonMonoxide_sensor_zenith_angle],
        'CarbonMonoxide_solar_azimuth_angle': [CarbonMonoxide_solar_azimuth_angle],
        'CarbonMonoxide_solar_zenith_angle': [CarbonMonoxide_solar_zenith_angle],
        'NitrogenDioxide_NO2_column_number_density': [NitrogenDioxide_NO2_column_number_density],
        'NitrogenDioxide_tropospheric_NO2_column_number_density': [NitrogenDioxide_tropospheric_NO2_column_number_density],
        'NitrogenDioxide_stratospheric_NO2_column_number_density': [NitrogenDioxide_stratospheric_NO2_column_number_density],
        'NitrogenDioxide_NO2_slant_column_number_density': [NitrogenDioxide_NO2_slant_column_number_density],
        'NitrogenDioxide_tropopause_pressure': [NitrogenDioxide_tropopause_pressure],
        'NitrogenDioxide_absorbing_aerosol_index': [NitrogenDioxide_absorbing_aerosol_index],
        'NitrogenDioxide_cloud_fraction': [NitrogenDioxide_cloud_fraction],
        'NitrogenDioxide_sensor_altitude': [NitrogenDioxide_sensor_altitude],
        'NitrogenDioxide_sensor_azimuth_angle': [NitrogenDioxide_sensor_azimuth_angle],
        'NitrogenDioxide_sensor_zenith_angle': [NitrogenDioxide_sensor_zenith_angle],
        'NitrogenDioxide_solar_azimuth_angle': [NitrogenDioxide_solar_azimuth_angle],
        'NitrogenDioxide_solar_zenith_angle': [NitrogenDioxide_solar_zenith_angle],
        'Formaldehyde_tropospheric_HCHO_column_number_density': [Formaldehyde_tropospheric_HCHO_column_number_density],
        'Formaldehyde_tropospheric_HCHO_column_number_density_amf': [Formaldehyde_tropospheric_HCHO_column_number_density_amf],
        'Formaldehyde_HCHO_slant_column_number_density': [Formaldehyde_HCHO_slant_column_number_density],
        'Formaldehyde_cloud_fraction': [Formaldehyde_cloud_fraction],
        'Formaldehyde_solar_zenith_angle': [Formaldehyde_solar_zenith_angle],
        'Formaldehyde_solar_azimuth_angle': [Formaldehyde_solar_azimuth_angle],
        'Formaldehyde_sensor_zenith_angle': [Formaldehyde_sensor_zenith_angle],
        'Formaldehyde_sensor_azimuth_angle': [Formaldehyde_sensor_azimuth_angle],
        'UvAerosolIndex_absorbing_aerosol_index': [UvAerosolIndex_absorbing_aerosol_index],
        'UvAerosolIndex_sensor_altitude': [UvAerosolIndex_sensor_altitude],
        'UvAerosolIndex_sensor_azimuth_angle': [UvAerosolIndex_sensor_azimuth_angle],
        'UvAerosolIndex_sensor_zenith_angle': [UvAerosolIndex_sensor_zenith_angle],
        'UvAerosolIndex_solar_azimuth_angle': [UvAerosolIndex_solar_azimuth_angle],
        'UvAerosolIndex_solar_zenith_angle': [UvAerosolIndex_solar_zenith_angle],
        'Ozone_O3_column_number_density': [Ozone_O3_column_number_density],
        'Ozone_O3_column_number_density_amf': [Ozone_O3_column_number_density_amf],
        'Ozone_O3_slant_column_number_density': [Ozone_O3_slant_column_number_density],
        'Ozone_O3_effective_temperature': [Ozone_O3_effective_temperature],
        'Ozone_cloud_fraction': [Ozone_cloud_fraction],
        'Ozone_sensor_azimuth_angle': [Ozone_sensor_azimuth_angle],
        'Ozone_sensor_zenith_angle': [Ozone_sensor_zenith_angle],
        'Ozone_solar_azimuth_angle': [Ozone_solar_azimuth_angle],
        'Ozone_solar_zenith_angle': [Ozone_solar_zenith_angle],
        'UvAerosolLayerHeight_aerosol_height': [UvAerosolLayerHeight_aerosol_height],
        'UvAerosolLayerHeight_aerosol_pressure': [UvAerosolLayerHeight_aerosol_pressure],
        'UvAerosolLayerHeight_aerosol_optical_depth': [UvAerosolLayerHeight_aerosol_optical_depth],
        'UvAerosolLayerHeight_sensor_zenith_angle': [UvAerosolLayerHeight_sensor_zenith_angle],
        'UvAerosolLayerHeight_sensor_azimuth_angle': [UvAerosolLayerHeight_sensor_azimuth_angle],
        'UvAerosolLayerHeight_solar_azimuth_angle': [UvAerosolLayerHeight_solar_azimuth_angle],
        'UvAerosolLayerHeight_solar_zenith_angle': [UvAerosolLayerHeight_solar_zenith_angle],
        'Cloud_cloud_fraction': [Cloud_cloud_fraction],
        'Cloud_cloud_top_pressure': [Cloud_cloud_top_pressure],
        'Cloud_cloud_top_height': [Cloud_cloud_top_height],
        'Cloud_cloud_base_pressure': [Cloud_cloud_base_pressure],
        'Cloud_cloud_base_height': [Cloud_cloud_base_height],
        'Cloud_cloud_optical_depth': [Cloud_cloud_optical_depth],
        'Cloud_surface_albedo': [Cloud_surface_albedo],
        'Cloud_sensor_azimuth_angle': [Cloud_sensor_azimuth_angle],
        'Cloud_sensor_zenith_angle': [Cloud_sensor_zenith_angle],
        'Cloud_solar_azimuth_angle': [Cloud_solar_azimuth_angle],
        'Cloud_solar_zenith_angle': [Cloud_solar_zenith_angle],
    })
    
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('CO2.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

def main():
    st.title("Sıcaklık Tahmin Uygulaması")
    st.write("Veri Girin")

    latitude = st.sidebar.number_input("Latitude", value=0.0, format="%.6f")
    longitude = st.sidebar.number_input("Longitude", value=0.0, format="%.6f")
    year = st.sidebar.number_input("Year", value=2024, step=1)
    week_no = st.sidebar.number_input("Week No", value=1, step=1)
    SulphurDioxide_SO2_column_number_density = st.sidebar.number_input("SO2 Column Number Density", value=0.0, format="%.6f")
    SulphurDioxide_SO2_column_number_density_amf = st.sidebar.number_input("SO2 Column Number Density AMF", value=0.0, format="%.6f")
    SulphurDioxide_SO2_slant_column_number_density = st.sidebar.number_input("SO2 Slant Column Number Density", value=0.0, format="%.6f")
    SulphurDioxide_cloud_fraction = st.sidebar.number_input("SO2 Cloud Fraction", value=0.0, format="%.6f")
    SulphurDioxide_sensor_azimuth_angle = st.sidebar.number_input("SO2 Sensor Azimuth Angle", value=0.0, format="%.6f")
    SulphurDioxide_sensor_zenith_angle = st.sidebar.number_input("SO2 Sensor Zenith Angle", value=0.0, format="%.6f")
    SulphurDioxide_solar_azimuth_angle = st.sidebar.number_input("SO2 Solar Azimuth Angle", value=0.0, format="%.6f")
    SulphurDioxide_solar_zenith_angle = st.sidebar.number_input("SO2 Solar Zenith Angle", value=0.0, format="%.6f")
    SulphurDioxide_SO2_column_number_density_15km = st.sidebar.number_input("SO2 Column Number Density 15km", value=0.0, format="%.6f")
    CarbonMonoxide_CO_column_number_density = st.sidebar.number_input("CO Column Number Density", value=0.0, format="%.6f")
    CarbonMonoxide_H2O_column_number_density = st.sidebar.number_input("H2O Column Number Density", value=0.0, format="%.6f")
    CarbonMonoxide_cloud_height = st.sidebar.number_input("CO Cloud Height", value=0.0, format="%.6f")
    CarbonMonoxide_sensor_altitude = st.sidebar.number_input("CO Sensor Altitude", value=0.0, format="%.6f")
    CarbonMonoxide_sensor_azimuth_angle = st.sidebar.number_input("CO Sensor Azimuth Angle", value=0.0, format="%.6f")
    CarbonMonoxide_sensor_zenith_angle = st.sidebar.number_input("CO Sensor Zenith Angle", value=0.0, format="%.6f")
    CarbonMonoxide_solar_azimuth_angle = st.sidebar.number_input("CO Solar Azimuth Angle", value=0.0, format="%.6f")
    CarbonMonoxide_solar_zenith_angle = st.sidebar.number_input("CO Solar Zenith Angle", value=0.0, format="%.6f")
    NitrogenDioxide_NO2_column_number_density = st.sidebar.number_input("NO2 Column Number Density", value=0.0, format="%.6f")
    NitrogenDioxide_tropospheric_NO2_column_number_density = st.sidebar.number_input("Tropospheric NO2 Column Number Density", value=0.0, format="%.6f")
    NitrogenDioxide_stratospheric_NO2_column_number_density = st.sidebar.number_input("Stratospheric NO2 Column Number Density", value=0.0, format="%.6f")
    NitrogenDioxide_NO2_slant_column_number_density = st.sidebar.number_input("NO2 Slant Column Number Density", value=0.0, format="%.6f")
    NitrogenDioxide_tropopause_pressure = st.sidebar.number_input("NO2 Tropopause Pressure", value=0.0, format="%.6f")
    NitrogenDioxide_absorbing_aerosol_index = st.sidebar.number_input("NO2 Absorbing Aerosol Index", value=0.0, format="%.6f")
    NitrogenDioxide_cloud_fraction = st.sidebar.number_input("NO2 Cloud Fraction", value=0.0, format="%.6f")
    NitrogenDioxide_sensor_altitude = st.sidebar.number_input("NO2 Sensor Altitude", value=0.0, format="%.6f")
    NitrogenDioxide_sensor_azimuth_angle = st.sidebar.number_input("NO2 Sensor Azimuth Angle", value=0.0, format="%.6f")
    NitrogenDioxide_sensor_zenith_angle = st.sidebar.number_input("NO2 Sensor Zenith Angle", value=0.0, format="%.6f")
    NitrogenDioxide_solar_azimuth_angle = st.sidebar.number_input("NO2 Solar Azimuth Angle", value=0.0, format="%.6f")
    NitrogenDioxide_solar_zenith_angle = st.sidebar.number_input("NO2 Solar Zenith Angle", value=0.0, format="%.6f")
    Formaldehyde_tropospheric_HCHO_column_number_density = st.sidebar.number_input("Tropospheric HCHO Column Number Density", value=0.0, format="%.6f")
    Formaldehyde_tropospheric_HCHO_column_number_density_amf = st.sidebar.number_input("Tropospheric HCHO Column Number Density AMF", value=0.0, format="%.6f")
    Formaldehyde_HCHO_slant_column_number_density = st.sidebar.number_input("HCHO Slant Column Number Density", value=0.0, format="%.6f")
    Formaldehyde_cloud_fraction = st.sidebar.number_input("HCHO Cloud Fraction", value=0.0, format="%.6f")
    Formaldehyde_solar_zenith_angle = st.sidebar.number_input("HCHO Solar Zenith Angle", value=0.0, format="%.6f")
    Formaldehyde_solar_azimuth_angle = st.sidebar.number_input("HCHO Solar Azimuth Angle", value=0.0, format="%.6f")
    Formaldehyde_sensor_zenith_angle = st.sidebar.number_input("HCHO Sensor Zenith Angle", value=0.0, format="%.6f")
    Formaldehyde_sensor_azimuth_angle = st.sidebar.number_input("HCHO Sensor Azimuth Angle", value=0.0, format="%.6f")
    UvAerosolIndex_absorbing_aerosol_index = st.sidebar.number_input("UV Aerosol Index Absorbing Aerosol Index", value=0.0, format="%.6f")
    UvAerosolIndex_sensor_altitude = st.sidebar.number_input("UV Aerosol Index Sensor Altitude", value=0.0, format="%.6f")
    UvAerosolIndex_sensor_azimuth_angle = st.sidebar.number_input("UV Aerosol Index Sensor Azimuth Angle", value=0.0, format="%.6f")
    UvAerosolIndex_sensor_zenith_angle = st.sidebar.number_input("UV Aerosol Index Sensor Zenith Angle", value=0.0, format="%.6f")
    UvAerosolIndex_solar_azimuth_angle = st.sidebar.number_input("UV Aerosol Index Solar Azimuth Angle", value=0.0, format="%.6f")
    UvAerosolIndex_solar_zenith_angle = st.sidebar.number_input("UV Aerosol Index Solar Zenith Angle", value=0.0, format="%.6f")
    Ozone_O3_column_number_density = st.sidebar.number_input("O3 Column Number Density", value=0.0, format="%.6f")
    Ozone_O3_column_number_density_amf = st.sidebar.number_input("O3 Column Number Density AMF", value=0.0, format="%.6f")
    Ozone_O3_slant_column_number_density = st.sidebar.number_input("O3 Slant Column Number Density", value=0.0, format="%.6f")
    Ozone_O3_effective_temperature = st.sidebar.number_input("O3 Effective Temperature", value=0.0, format="%.6f")
    Ozone_cloud_fraction = st.sidebar.number_input("O3 Cloud Fraction", value=0.0, format="%.6f")
    Ozone_sensor_azimuth_angle = st.sidebar.number_input("O3 Sensor Azimuth Angle", value=0.0, format="%.6f")
    Ozone_sensor_zenith_angle = st.sidebar.number_input("O3 Sensor Zenith Angle", value=0.0, format="%.6f")
    Ozone_solar_azimuth_angle = st.sidebar.number_input("O3 Solar Azimuth Angle", value=0.0, format="%.6f")
    Ozone_solar_zenith_angle = st.sidebar.number_input("O3 Solar Zenith Angle", value=0.0, format="%.6f")
    UvAerosolLayerHeight_aerosol_height = st.sidebar.number_input("Aerosol Height", value=0.0, format="%.6f")
    UvAerosolLayerHeight_aerosol_pressure = st.sidebar.number_input("Aerosol Pressure", value=0.0, format="%.6f")
    UvAerosolLayerHeight_aerosol_optical_depth = st.sidebar.number_input("Aerosol Optical Depth", value=0.0, format="%.6f")
    UvAerosolLayerHeight_sensor_zenith_angle = st.sidebar.number_input("Aerosol Sensor Zenith Angle", value=0.0, format="%.6f")
    UvAerosolLayerHeight_sensor_azimuth_angle = st.sidebar.number_input("Aerosol Sensor Azimuth Angle", value=0.0, format="%.6f")
    UvAerosolLayerHeight_solar_zenith_angle = st.sidebar.number_input("Aerosol Solar Zenith Angle", value=0.0, format="%.6f")
    UvAerosolLayerHeight_solar_azimuth_angle = st.sidebar.number_input("Aerosol Solar Azimuth Angle", value=0.0, format="%.6f")
    Cloud_cloud_fraction = st.sidebar.number_input("Cloud Fraction", value=0.0, format="%.6f")
    Cloud_cloud_top_pressure = st.sidebar.number_input("Cloud Top Pressure", value=0.0, format="%.6f")
    Cloud_cloud_top_height = st.sidebar.number_input("Cloud Top Height", value=0.0, format="%.6f")
    Cloud_cloud_base_pressure = st.sidebar.number_input("Cloud Base Pressure", value=0.0, format="%.6f")
    Cloud_cloud_base_height = st.sidebar.number_input("Cloud Base Height", value=0.0, format="%.6f")
    Cloud_cloud_optical_depth = st.sidebar.number_input("Cloud Optical Depth", value=0.0, format="%.6f")
    Cloud_surface_albedo = st.sidebar.number_input("Cloud Surface Albedo", value=0.0, format="%.6f")
    Cloud_sensor_azimuth_angle = st.sidebar.number_input("Cloud Sensor Azimuth Angle", value=0.0, format="%.6f")
    Cloud_sensor_zenith_angle = st.sidebar.number_input("Cloud Sensor Zenith Angle", value=0.0, format="%.6f")
    Cloud_solar_azimuth_angle = st.sidebar.number_input("Cloud Solar Azimuth Angle", value=0.0, format="%.6f")
    Cloud_solar_zenith_angle = st.sidebar.number_input("Cloud Solar Zenith Angle", value=0.0, format="%.6f")

    if st.button('Tahmin Et'):
        time = time_pred(latitude,longitude,year,week_no,SulphurDioxide_SO2_column_number_density,SulphurDioxide_SO2_column_number_density_amf,
                         SulphurDioxide_SO2_slant_column_number_density,SulphurDioxide_cloud_fraction,SulphurDioxide_sensor_azimuth_angle,
                         SulphurDioxide_sensor_zenith_angle,SulphurDioxide_solar_azimuth_angle,SulphurDioxide_solar_zenith_angle,
                         SulphurDioxide_SO2_column_number_density_15km,CarbonMonoxide_CO_column_number_density,CarbonMonoxide_H2O_column_number_density,
                         CarbonMonoxide_cloud_height,CarbonMonoxide_sensor_altitude,CarbonMonoxide_sensor_azimuth_angle,CarbonMonoxide_sensor_zenith_angle,
                         CarbonMonoxide_solar_azimuth_angle,CarbonMonoxide_solar_zenith_angle,NitrogenDioxide_NO2_column_number_density,
                         NitrogenDioxide_tropospheric_NO2_column_number_density,NitrogenDioxide_stratospheric_NO2_column_number_density,
                         NitrogenDioxide_NO2_slant_column_number_density,NitrogenDioxide_tropopause_pressure,NitrogenDioxide_absorbing_aerosol_index,
                         NitrogenDioxide_cloud_fraction,NitrogenDioxide_sensor_altitude,NitrogenDioxide_sensor_azimuth_angle,
                         NitrogenDioxide_sensor_zenith_angle,NitrogenDioxide_solar_azimuth_angle,NitrogenDioxide_solar_zenith_angle,
                         Formaldehyde_tropospheric_HCHO_column_number_density,Formaldehyde_tropospheric_HCHO_column_number_density_amf,
                         Formaldehyde_HCHO_slant_column_number_density,Formaldehyde_cloud_fraction,Formaldehyde_solar_zenith_angle,
                         Formaldehyde_solar_azimuth_angle,Formaldehyde_sensor_zenith_angle,Formaldehyde_sensor_azimuth_angle,UvAerosolIndex_absorbing_aerosol_index,
                         UvAerosolIndex_sensor_altitude,UvAerosolIndex_sensor_azimuth_angle,UvAerosolIndex_sensor_zenith_angle,UvAerosolIndex_solar_azimuth_angle,
                         UvAerosolIndex_solar_zenith_angle,Ozone_O3_column_number_density,Ozone_O3_column_number_density_amf,Ozone_O3_slant_column_number_density,
                         Ozone_O3_effective_temperature,Ozone_cloud_fraction,Ozone_sensor_azimuth_angle,Ozone_sensor_zenith_angle,Ozone_solar_azimuth_angle,
                         Ozone_solar_zenith_angle,UvAerosolLayerHeight_aerosol_height,UvAerosolLayerHeight_aerosol_pressure,UvAerosolLayerHeight_aerosol_optical_depth,
                         UvAerosolLayerHeight_sensor_zenith_angle,UvAerosolLayerHeight_sensor_azimuth_angle,UvAerosolLayerHeight_solar_azimuth_angle,
                         UvAerosolLayerHeight_solar_zenith_angle,Cloud_cloud_fraction,Cloud_cloud_top_pressure,Cloud_cloud_top_height,Cloud_cloud_base_pressure,
                         Cloud_cloud_base_height,Cloud_cloud_optical_depth,Cloud_surface_albedo,Cloud_sensor_azimuth_angle,Cloud_sensor_zenith_angle,
                         Cloud_solar_azimuth_angle,Cloud_solar_zenith_angle)
        st.write(f'Tahmin edilen sıcaklık: {time:.2f} °C')

if __name__ == '__main__':
    main()
