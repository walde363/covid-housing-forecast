import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import datetime
from google import genai

def get_ai_summary(data_table_md, level_name):
    """Invokes Gemini to analyze the metrics table."""
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            return "⚠️ Gemini API Key not found in secrets. Please add it to continue."
    except Exception:
        return "⚠️ Streamlit secrets file not found. Please create .streamlit/secrets.toml."

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    prompt = f"""
    You are a Senior Real Estate Data Scientist. Analyze the following model performance table for housing price forecasts at the {level_name} level.
    
    Performance Data (Markdown):
    {data_table_md}
    
    Your task:
    1. Identify the 'Winner' model(s) based on RMSE and MASE.
    2. Compare the advanced models (XGBoost, RF, Prophet, SARIMAX) against the 'Seasonal Naive' baseline. 
    3. Note if any model has a MASE > 1 (meaning it is worse than the baseline).
    4. Provide a brief (3-4 sentences) executive summary of which model is safest for investment decisions.
    
    Keep the tone professional, data-driven, and concise.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"

def generate_pdf_report(all_results):
    """Generates a comprehensive PDF report covering US, State, and Region levels."""
    pdf = FPDF()

    def add_table_section(title, data_list):
        if not data_list:
            return
        
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 16)
        pdf.cell(0, 10, "Housing Forecast Comparison Report", 0, 1, 'C')
        pdf.set_font("helvetica", 'I', 10)
        pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("helvetica", 'B', 14)
        pdf.cell(0, 10, f"Analysis Level: {title}", 0, 1, 'L')
        pdf.ln(5)

        df = pd.DataFrame(data_list)
        pivot_df = df.pivot(index="Model", columns="Metric", values="Value")
        
        # Table Header
        pdf.set_font("helvetica", 'B', 9)
        metrics = pivot_df.columns.tolist()
        col_widths = [55] + [(135 / len(metrics))] * len(metrics)
        
        pdf.cell(col_widths[0], 10, "Model", 1, 0, 'C')
        for i, metric in enumerate(metrics):
            pdf.cell(col_widths[i+1], 10, metric, 1, 0, 'C')
        pdf.ln()
        
        # Table Data
        pdf.set_font("helvetica", '', 8)
        for model, row in pivot_df.iterrows():
            pdf.cell(col_widths[0], 8, str(model)[:35], 1)
            for i, metric in enumerate(metrics):
                val = row[metric]
                val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                pdf.cell(col_widths[i+1], 8, val_str, 1, 0, 'C')
            pdf.ln()

    # Prepare data for each level
    levels = ["us", "state", "region"]
    for level in levels:
        level_data = []
        for model_name, res in all_results.items():
            current_metrics = res.get(level) if isinstance(res, dict) else None
            
            if level == "region" and isinstance(current_metrics, dict):
                for region_name, metrics_list in current_metrics.items():
                    for m in metrics_list:
                        level_data.append({
                            "Model": f"{model_name} ({region_name})",
                            "Metric": m["Metric"],
                            "Value": m["Value"]
                        })
            elif isinstance(current_metrics, list):
                for m in current_metrics:
                    level_data.append({
                        "Model": model_name,
                        "Metric": m["Metric"],
                        "Value": m["Value"]
                    })
        
        add_table_section(level.upper(), level_data)

    return bytes(pdf.output())

def render_comparison(all_results):
    st.header("🏆 Model Comparison")
    
    # Export Full Report Button at the top
    try:
        full_pdf_data = generate_pdf_report(all_results)
        st.download_button(
            label="📥 Download Full Analysis Report (US, State, Regions)",
            data=full_pdf_data,
            file_name=f"housing_forecast_full_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

    # Add level selection to compare across different scales
    comparison_level = st.radio("Select Comparison Level", ["Region", "State", "US"], horizontal=True)
    level_key = comparison_level.lower()

    st.markdown(f"""
    This section compares the performance of all models for the **{comparison_level}-level** forecasts. 
    Lower values are better for RMSE, MAE, MAPE, and MASE, while higher values are better for R².
    """)
    
    comparison_data = []
    for model_name, res in all_results.items():
        # Handle nested dictionary results or fallback for models only supporting region (like SNaive)
        current_metrics = None
        if isinstance(res, dict):
            current_metrics = res.get(level_key)

        if level_key == "region" and isinstance(current_metrics, dict):
            # current_metrics is {region_name: metrics_list}
            for region_name, metrics_list in current_metrics.items():
                for m in metrics_list:
                    comparison_data.append({
                        "Model": f"{model_name} ({region_name})",
                        "Metric": m["Metric"],
                        "Value": m["Value"]
                    })
        elif isinstance(current_metrics, list):
            for m in current_metrics:
                comparison_data.append({
                    "Model": model_name,
                    "Metric": m["Metric"],
                    "Value": m["Value"]
                })
    
    if not comparison_data:
        st.warning("No metrics available. Please visit the individual model tabs to generate results.")
        return

    df = pd.DataFrame(comparison_data)
    
    # Summary Table
    st.subheader("📊 Metrics Summary Table")
    
    pivot_df = df.pivot(index="Model", columns="Metric", values="Value")

    # AI Analysis Section
    st.divider()
    st.subheader("✨ AI Insights")
    if st.button("🚀 Generate AI Summary", help="Uses Gemini to analyze the current comparison table"):
        with st.spinner("Analyzing performance data..."):
            # Convert table to markdown for the LLM
            table_md = pivot_df.to_markdown()
            analysis = get_ai_summary(table_md, comparison_level)
            st.info(analysis)

    
    def highlight_best(s):
        if s.name == 'R2':
            is_best = s == s.max()
        else:
            is_best = s == s.min()
        return ['background-color: #0d47a1' if v else '' for v in is_best]

    st.dataframe(pivot_df.style.apply(highlight_best), width='stretch')
    st.caption("Blue cells indicate the best performing model for that specific metric.")

    # Visual Comparison
    st.divider()
    st.subheader("📈 Performance Visualization")
    
    metrics = ["RMSE", "MAE", "MAPE", "MASE", "R2"]
    selected_metric = st.selectbox("Select Metric to Visualize", [m for m in metrics if m in df["Metric"].unique()])
    
    metric_df = df[df["Metric"] == selected_metric].sort_values("Value", ascending=(selected_metric != "R2"))
    fig = px.bar(
        metric_df,
        x="Model",
        y="Value",
        color="Model",
        text_auto='.3f',
        template="plotly_dark",
        title=f"Comparison of {selected_metric} across models"
    )
    st.plotly_chart(fig, width='stretch')