import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Purple + White Table Styling
# ===============================
def inject_table_css():
    st.markdown("""
    <style>
        .table-wrapper {
            max-width: 100%;
            overflow-x: auto;
            margin: 16px auto;
        }

        table.custom-table {
            border-collapse: collapse;
            font-size: 14px;
            font-family: Inter, system-ui, sans-serif;
            white-space: nowrap;
            margin: 0 auto;
        }

        table.custom-table th {
            background-color: #5b5fe8;
            color: white !important;
            font-weight: 700;
            padding: 8px 16px;
            text-align: center;
        }

        table.custom-table td {
            background-color: #f7f8ff;
            color: #1f2937;
            padding: 8px 16px;
            text-align: center;
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """, unsafe_allow_html=True)


# ===============================
# Render Styled Table
# ===============================
def render_table(df):
    html = df.to_html(index=False, classes="custom-table", border=0)
    st.markdown(f"<div class='table-wrapper'>{html}</div>", unsafe_allow_html=True)


# ===============================
# Preprocessing Page
# ===============================
def preprocessing_page():
    inject_table_css()

    st.header("🛠️ Preprocessing Stage")

    if "data" not in st.session_state:
        st.warning("⚠️ Upload dataset first.")
        return

    df = st.session_state["data"]

    # =========================
    # Column-wise Data Quality Summary
    # =========================
    st.subheader("📌 Column-wise Data Quality Summary")

    summary = []

    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isnull().sum()

        outliers = 0
        if pd.api.types.is_numeric_dtype(col_data):
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower) | (col_data > upper)].count()

        summary.append({
            "Column Name": col,
            "Data Type": col_data.dtype,
            "Missing Values": missing_count,
            "Outliers (IQR)": outliers,
            "Unique Values": col_data.nunique()
        })

    render_table(pd.DataFrame(summary))

    st.divider()

    # =========================
    # Duplicate Records
    # =========================
    st.subheader("🧬 Duplicate Records")

    dup_count = df.duplicated().sum()

    if dup_count > 0:
        render_table(pd.DataFrame({
            "Metric": ["Duplicate Rows Found"],
            "Value": [dup_count]
        }))
        render_table(df[df.duplicated()])
    else:
        render_table(pd.DataFrame({"Status": ["No duplicate rows found"]}))

    st.divider()

    # =========================
    # Outlier Inspection (Toggle Boxplots)
    # =========================
    if "show_boxplots" not in st.session_state:
        st.session_state.show_boxplots = False

    if st.button("📦 Boxplots for Numerical Columns"):
        st.session_state.show_boxplots = not st.session_state.show_boxplots

    if st.session_state.show_boxplots:
        st.subheader("📦 Outlier Inspection (Numerical Features)")

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) == 0:
            st.info("No numerical columns available for boxplot analysis.")
        else:
            cols_per_row = 4
            rows = [
                num_cols[i:i + cols_per_row]
                for i in range(0, len(num_cols), cols_per_row)
            ]

            for row in rows:
                col_containers = st.columns(cols_per_row)

                for idx, col_name in enumerate(row):
                    with col_containers[idx]:
                        st.markdown(
                            f"<p style='text-align:center; font-weight:600; font-size:13px;'>{col_name}</p>",
                            unsafe_allow_html=True
                        )
                        fig, ax = plt.subplots(figsize=(2.2, 2.2))
                        ax.boxplot(df[col_name].dropna(), vert=True)
                        ax.set_xticks([])
                        st.pyplot(fig)
