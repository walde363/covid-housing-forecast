import streamlit as st

def metrics_display(metrics):
    st.markdown("""
                <div style="display:flex; gap:20px; align-items:center; margin-top:10px; margin-bottom:10px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div style="width:16px; height:16px; background-color:#34C6F4; border-radius:3px;"></div>
                        <span>Very Good</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div style="width:16px; height:16px; background-color:#A7DFF3; border-radius:3px;"></div>
                        <span>Good</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div style="width:16px; height:16px; background-color:#FF5C00; border-radius:3px;"></div>
                        <span>Bad</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    st.header("Model Metrics")
    for item in metrics:
            value = f"{item['Value']:.4f}" if isinstance(item["Value"], (int, float)) else item["Value"]

            st.markdown(
                f"<h2>{item['Metric']}: <span style='color:{item['Color']};'>{value}</span></h2>",
                unsafe_allow_html=True
                )