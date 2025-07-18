import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- 1. Define the Lactation Curve Model (Wood's Model) ---
def woods_model(t, a, b, c):
    """
    Wood's Incomplete Gamma Function for lactation curves.
    Y(t) = a * t^b * e^(-c*t)
    - t: Days in Milk (DIM)
    - a: Scaling factor related to overall yield
    - b: Factor for the inclining slope (pre-peak)
    - c: Factor for the declining slope (post-peak)
    """
    # Add a small epsilon to t to avoid log(0) or 0^b issues if t contains 0
    epsilon = 1e-9
    return a * (t + epsilon)**b * np.exp(-c * (t + epsilon))

# --- 2. Streamlit App Interface ---
st.set_page_config(layout="wide")

st.title("🐮 Lactation Curve Analysis for Dairy Animals")
st.markdown("""
This application analyzes milk production data using the **Wood's Lactation Model** to calculate key performance indicators for a dairy animal's lactation cycle.
""")

# --- 3. Sidebar for Inputs and Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    lactation_length = st.number_input("Standard Lactation Length (days)", min_value=100, max_value=500, value=305, step=5)
    
    st.header("📋 Input Your Data")
    st.markdown("Paste your data below as `Day,Yield`. Each entry should be on a new line.")
    
    # Example data to guide the user
    default_data = """15,25.5
30,35.1
45,40.2
60,42.5
75,41.8
90,40.1
120,38.5
150,36.2
180,34.0
210,31.5
240,29.1
270,26.8
300,24.5"""

    data_input = st.text_area("Milk Yield Data (Day, Yield)", value=default_data, height=300)

# --- 4. Main Panel for Analysis and Results ---
if st.button("📈 Analyze Lactation Curve"):
    try:
        # --- Data Parsing and Validation ---
        lines = data_input.strip().split('\n')
        if len(lines) < 5:
            st.error("Please provide at least 5 data points for an accurate analysis.")
        else:
            data = []
            for line in lines:
                day, yield_val = line.split(',')
                data.append([int(day), float(yield_val)])
            
            df = pd.DataFrame(data, columns=['DIM', 'Yield'])
            days_in_milk = df['DIM'].values
            milk_yield = df['Yield'].values

            # --- Model Fitting ---
            # Provide initial guesses for parameters a, b, c to help the algorithm
            initial_guesses = [15, 0.2, 0.003]
            popt, pcov = curve_fit(woods_model, days_in_milk, milk_yield, p0=initial_guesses, maxfev=10000)
            a, b, c = popt

            # --- Calculate Key Performance Indicators (KPIs) ---
            # Time to Peak (in days) is where the derivative is zero: t = b/c
            time_to_peak = b / c
            # Peak Yield is the model's output at the time to peak
            peak_yield = woods_model(time_to_peak, a, b, c)
            
            # Total Yield over the standard lactation period
            total_yield, _ = quad(woods_model, 1, lactation_length, args=(a, b, c))
            
            # Persistency: (Yield at day 250 / Peak Yield) * 100
            yield_at_250 = woods_model(250, a, b, c)
            persistency = (yield_at_250 / peak_yield) * 100 if peak_yield > 0 else 0

            # --- Display Results ---
            st.success("✅ Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Key Performance Indicators")
                st.metric(label="Peak Daily Yield", value=f"{peak_yield:.2f} kg/day")
                st.metric(label="Time to Peak Yield", value=f"{time_to_peak:.1f} days")
                st.metric(label=f"Total {lactation_length}-Day Yield", value=f"{total_yield:.0f} kg")
                st.metric(label="Lactation Persistency", value=f"{persistency:.1f} %", help="A measure of how well milk production is maintained after the peak. Higher is generally better.")

            with col2:
                st.subheader("🔬 Model Parameters")
                st.info(f"""
                The analysis is based on the Wood's Model: **Y(t) = a * t^b * e^(-ct)**
                
                - **Parameter 'a'**: {a:.4f} (Initial yield scaling factor)
                - **Parameter 'b'**: {b:.4f} (Pre-peak incline rate)
                - **Parameter 'c'**: {c:.4f} (Post-peak decline rate)
                """)

            # --- Generate and Display Plot ---
            st.subheader("📈 Lactation Curve Visualization")
            
            t_smooth = np.linspace(1, lactation_length, 305)
            y_smooth = woods_model(t_smooth, a, b, c)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(days_in_milk, milk_yield, label='Actual Data Points', color='blue', zorder=5)
            ax.plot(t_smooth, y_smooth, label="Fitted Wood's Curve", color='red', linewidth=2)
            ax.axvline(time_to_peak, color='green', linestyle='--', label=f'Peak at {time_to_peak:.1f} days')
            ax.axhline(peak_yield, color='orange', linestyle='--', label=f'Peak Yield: {peak_yield:.2f} kg')
            
            ax.set_title('Lactation Curve Analysis', fontsize=16)
            ax.set_xlabel('Days in Milk (DIM)', fontsize=12)
            ax.set_ylabel('Daily Milk Yield (kg)', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure your data is formatted correctly (e.g., `30,25.5`) with each entry on a new line and contains enough points for analysis.")