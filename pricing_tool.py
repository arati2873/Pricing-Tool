 
# Define SKU limits
BASIC_SKU_LIMIT = 300000
IS_PRO_VERSION = False  # Set to True for Pro version

    
# --- Streamlit App: Price Revision Tool ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

#Stage 5 Complete:Product Group and family level summary with override option
# Set up page
st.set_page_config(page_title="Price Revision Tool", layout="wide")
st.title("📈 Intelligent Price Revision Tool")

st.info("🔒 Your data is not stored or shared. Files are processed securely within your session for analysis only.")

st.sidebar.markdown("[🛒 Buy Access - $99](https://yadavarati.gumroad.com/l/IntelligentPriceRevisionTool)")

# --- Access Gate ---
ACCESS_CODE = "A"

with st.expander("🔐 Enter Access Code to Unlock the Tool"):
    user_code = st.text_input("Access Code", type="password")

if user_code != ACCESS_CODE:
    st.warning("This is a premium tool. Please enter a valid access code to continue.")
    st.stop()


with st.expander("❓ How to Use This Tool (Click to Expand)"):
    st.markdown("""
    ### 📘 Required Inputs
    This tool requires 6 CSV files:
    1. **cost_file.csv** – Cost data by SKU
    2. **sales_data_1.csv** – Recent sales data (e.g., last 6 months or 1 year)
    3. **sales_data_2.csv** – Previous period sales (e.g., 6 months before `sales_data_1`)
    4. **standard_selling_price.csv** – Current prices by SKU
    5. **monthly_sales.csv** – Monthly sales quantity and ASP by SKU
    6. **product_classification.csv** – Classification mapping (Family, Group, etc.)

    ➡️ Make sure **column headers are not changed** from the provided templates.

    ---

    ### ⚙️ How It Works
    - Upload all required files from the sidebar
    - Adjust scoring weights if needed
    - Set global or override % increase
    - Download SKU-level recommended pricing

    ---

    📂 You can [download sample input files here](https://github.com/arati2873/Pricing-Tool/tree/main/Sample%20data)

    📄 Full user guide available in the [README](https://github.com/arati2873/Pricing-Tool/blob/main/README.md)
    """)



# Step 1: Upload files
st.sidebar.markdown("### 📤 Upload Required Files")

uploaded_files = {
    "Cost File": st.sidebar.file_uploader("Upload cost_file.csv", type="csv"),
    "Sales Data 1": st.sidebar.file_uploader("Upload sales_data_1.csv", type="csv"),
    "Sales Data 2": st.sidebar.file_uploader("Upload sales_data_2.csv", type="csv"),
    "Price Today": st.sidebar.file_uploader("Upload standard_selling_price.csv", type="csv"),
    "Product Classification": st.sidebar.file_uploader("Upload product_classification.csv", type="csv")
}

if all(uploaded_files.values()):
    data_loaded = True
    file_paths = uploaded_files
else:
    #st.warning("⚠️ Please upload all five input files to continue.")
    data_loaded = False

    
# --- Helper Functions ---
def normalize_scores(df, score_col):
    min_score, max_score = df[score_col].min(), df[score_col].max()
    return (df[score_col] - min_score) / (max_score - min_score) if max_score > min_score else 0

def apply_override(df, key_col, selected_keys, pct_dict, score_col, price_col, revenue_col, asp_col):
    for key, pct in pct_dict.items():
        mask = df[key_col] == key
        subset = df[mask].copy()
        if subset[score_col].sum() == 0:
            df.loc[mask, 'Assigned_Price_Increase_%'] = pct
        else:
            norm_score = subset[score_col] / subset[score_col].sum()
            subset['Estimated_Qty'] = subset[revenue_col] / subset[asp_col]
            subset['Base_New_Price'] = subset[price_col] * (1 + norm_score * pct / 100)
            subset['Base_New_Revenue'] = subset[revenue_col] * (1 + norm_score * pct / 100)
            total_old = subset[revenue_col].sum()
            target_new = total_old * (1 + pct / 100)
            actual_new = subset['Base_New_Revenue'].sum()
            multiplier = target_new / actual_new if actual_new > 0 else 1
            subset['Final_New_Price'] = subset['Base_New_Price'] * multiplier
            subset['Assigned_Price_Increase_%'] = (subset['Final_New_Price'] / subset[price_col] - 1) * 100
            df.loc[mask, 'Assigned_Price_Increase_%'] = subset['Assigned_Price_Increase_%'].values

    return df

    sku_count = df['SKU'].nunique()

    if not IS_PRO_VERSION and sku_count > BASIC_SKU_LIMIT:
        st.error(f"❌ You've exceeded the 30,000 SKU limit. Please upgrade to the Pro version.")
        st.stop()

def apply_global_score_increase(df, excluded_mask, score_col, target_pct):
    sub_df = df[excluded_mask]
    if sub_df[score_col].sum() == 0:
        df.loc[excluded_mask, 'Assigned_Price_Increase_%'] = target_pct
    else:
        weight = sub_df[score_col]
        multiplier = target_pct / weight.mean()
        df.loc[excluded_mask, 'Assigned_Price_Increase_%'] = weight * multiplier
    return df

def adjust_remaining(df, overridden_mask, target_total_revenue):
    override_new = df.loc[overridden_mask, 'New_Revenue'].sum()
    override_old = df.loc[overridden_mask, 'Revenue_1'].sum()
    remaining_target = target_total_revenue - override_new
    remaining_old = df['Revenue_1'].sum() - override_old
    if remaining_old > 0:
        actual_remaining_new = df.loc[~overridden_mask, 'New_Revenue'].sum()
        multiplier = remaining_target / actual_remaining_new if actual_remaining_new > 0 else 1
        df.loc[~overridden_mask, 'New_Price'] *= multiplier
        df.loc[~overridden_mask, 'Assigned_Price_Increase_%'] = (
            df.loc[~overridden_mask, 'New_Price'] / df.loc[~overridden_mask, 'Price_Today'] - 1
        ) * 100
    return df

def summarize_revenue(df, group_col):
    summary = df.groupby(group_col).agg(
        Total_Revenue_Old=('Revenue_1', 'sum'),
        Total_Revenue_New=('New_Revenue', 'sum'),
        TTL_Cost=('TTL_Cost', 'sum'),
        New_Cost=('New_Cost', 'sum')
    ).reset_index()

    summary['Revenue_Increase_%'] = (
        (summary['Total_Revenue_New'] - summary['Total_Revenue_Old']) / summary['Total_Revenue_Old']
    ) * 100

    summary['Cost_Increase_%'] = (
        (summary['New_Cost'] - summary['TTL_Cost']) / summary['TTL_Cost']
    ) * 100

    summary['Old_GM'] = summary['Total_Revenue_Old'] - summary['TTL_Cost']
    summary['New_GM'] = summary['Total_Revenue_New'] - summary['New_Cost']
    summary['GM_Impact'] = summary['New_GM'] - summary['Old_GM']

    return summary


# --- Main Logic ---
# 🧼 Sanitize all inputs
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

def clean_sku_column(df):
    df['SKU'] = df['SKU'].astype(str).str.strip().str.upper()
    return df

def clean_numeric_column(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

if data_loaded:
    cost_df = clean_column_names(pd.read_csv(file_paths["Cost File"]))
    sales_1 = clean_column_names(pd.read_csv(file_paths["Sales Data 1"]))
    sales_2 = clean_column_names(pd.read_csv(file_paths["Sales Data 2"]))
    price_today = clean_column_names(pd.read_csv(file_paths["Price Today"]))
    product_class = clean_column_names(pd.read_csv(file_paths["Product Classification"]))

    # 🧼 Clean SKU
    for df in [cost_df, sales_1, sales_2, price_today, product_class]:
        df = clean_sku_column(df)

    # ✅ Merge
    df = sales_1.merge(sales_2, on='SKU', how='left', suffixes=('_1', '_2'))
    df = df.merge(cost_df, on='SKU', how='left')
    df = df.merge(price_today, on='SKU', how='left')
    df = df.merge(product_class, on='SKU', how='left')

    # 🧼 Ensure numeric
    #numeric_cols = ['Revenue_1', 'Revenue_2', 'GM%_1', 'GM%_2', 'GM_1', 'GM_2', 'ASP_1', 'ASP_2','TTL_Cost','Qty','Cost_per_Unit']
    #for col in numeric_cols:
     #   df = clean_numeric_column(df, col)

    # ✅ Calculations
    df['Sales_Growth_%'] = ((df['Revenue_1'] - df['Revenue_2']) / df['Revenue_2']) * 100
    df['GM%_Change'] = df['GM%_1'] - df['GM%_2']
    df['Price_Change_%'] = ((df['ASP_1'] - df['ASP_2']) / df['ASP_2']) * 100
    df['GM_Abs_Change'] = df['GM_1'] - df['GM_2']
    # Calculate % change in Qty and ASP
    
    df['Qty_1'] = pd.to_numeric(df['Qty_1'], errors='coerce')
    df['Qty_2'] = pd.to_numeric(df['Qty_2'], errors='coerce')

    
    df['Qty_Change_%'] = ((df['Qty_1'] - df['Qty_2']) / df['Qty_2'].replace(0, np.nan)) * 100
    df['ASP_Change_%'] = ((df['ASP_1'] - df['ASP_2']) / df['ASP_2'].replace(0, np.nan)) * 100

    # Handle NaNs or inf values
    df['Qty_Change_%'] = df['Qty_Change_%'].replace([np.inf, -np.inf], 0).fillna(0)
    df['ASP_Change_%'] = df['ASP_Change_%'].replace([np.inf, -np.inf], 0).fillna(0)



    df['Revenue_1'] = df['Revenue_1'].fillna(0)
    df['Revenue_2'] = df['Revenue_2'].fillna(0)
    df['ASP_1'] = df['ASP_1'].replace(0, np.nan)
    df['ASP_2'] = df['ASP_2'].replace(0, np.nan)
    df['GM%_1'] = df['GM%_1'].fillna(0)
    df['GM%_2'] = df['GM%_2'].fillna(0)
    df['GM_1'] = df['GM_1'].fillna(0)
    df['GM_2'] = df['GM_2'].fillna(0)
    df['Cost_Change_%'] = df['Cost_Change_%'].fillna(0)
    df['Cost_Per_Unit_1'] = df['Cost_Per_Unit_1'].fillna(0)
    df['Cost_Per_Unit_2'] = df['Cost_Per_Unit_2'].fillna(0)
    df['TTL_Cost'] = df['TTL_Cost'].fillna(0)
    df['Qty_1'] = df['Qty_1'].fillna(0)
    df['Qty_2'] = df['Qty_2'].fillna(0)
    df['ASP_1'] = df['ASP_1'].fillna(0)
    df['ASP_2'] = df['ASP_2'].fillna(0)
    df['Sales_Growth_%'] = df['Sales_Growth_%'].fillna(0)



    # Show preview
    #st.subheader("✅ Merged & Calculated Data Sample")
    #st.dataframe(df.head(10))


    def scale_score(series): return MinMaxScaler(feature_range=(1, 15)).fit_transform(series.values.reshape(-1, 1)).flatten()
    def scale_score_inverse(series): return 16 - scale_score(series)

    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    def scale_familywise(df, column, inverse=False):
        df_result = df.copy()
        df_result[f'Score_{column}'] = np.nan

        for family in df_result['Product_Family'].unique():
            mask = df_result['Product_Family'] == family
            values = df_result.loc[mask, column].astype(float)

            # Replace NaNs and infs with median of valid values
            valid_values = values.replace([np.inf, -np.inf], np.nan)
            median_val = valid_values.median()
            filled_values = valid_values.fillna(median_val).values.reshape(-1, 1)

            try:
                scaler = MinMaxScaler()
                scaled_vals = scaler.fit_transform(filled_values).flatten()

                if inverse:
                    scaled_vals = 1 - scaled_vals  # Invert the score

                df_result.loc[mask, f'Score_{column}'] = scaled_vals
            except Exception as e:
                print(f"❌ Error scaling Product_Family '{family}': {e}")
                continue

        return df_result[f'Score_{column}']



    if 'Cost_Change_%' in df.columns and df['Cost_Change_%'].notna().any():
        df['Score_Cost_Change'] = scale_familywise(df, 'Cost_Change_%')
    else:
        df['Score_Cost_Change'] = 0
        
    if 'Sales_Growth_%' in df.columns and df['Sales_Growth_%'].notna().any():
        df['Score_Sales_Growth'] = scale_familywise(df, 'Sales_Growth_%')
    else:
        df['Score_Sales_Growth'] = 0
    df['Score_GM_Change'] = scale_familywise(df, 'GM%_Change')
    #df['Score_Elasticity'] = scale_familywise(df, 'Elasticity', inverse=True)
    df['Score_GM_Abs_Change'] = scale_familywise(df, 'GM_Abs_Change')
    df['Score_Qty_Change'] = scale_familywise(df, 'Qty_Change_%')
    df['Score_ASP_Change'] = scale_familywise(df, 'ASP_Change_%')


    # 2. Use a visible checkbox toggle with label
    show_weights = st.sidebar.checkbox("Show Advanced Score Weight Settings", value=True)

    # 3. Define presets
    preset_options = {
        "Balanced (Default)": {"Sales_Growth": 10, "Cost_Change": 10, "GM%_Change": 10, "Qty_Change": 10,"ASP_Change": 10, "GM_Abs_Change": 10},
        "Aggressive (Push ASP)": {"Sales_Growth": 20, "Cost_Change": 10, "GM%_Change": 15, "Qty_Change": 50,"ASP_Change": 30, "GM_Abs_Change": 15},
        "Defensive (Protect GM)": {"Sales_Growth": 30, "Cost_Change": 25, "GM%_Change": 25, "Qty_Change": 10,"ASP_Change": 30, "GM_Abs_Change": 10},
    }

    # 4. Session state to store current weights
    if 'weight_state' not in st.session_state:
        st.session_state.weight_state = preset_options["Balanced (Default)"].copy()

    if show_weights:
        # Preset selector
        selected_preset = st.sidebar.selectbox("Choose a Preset", list(preset_options.keys()))

        if st.sidebar.button("🔁 Apply Preset"):
            st.session_state.weight_state = preset_options[selected_preset].copy()

        if st.sidebar.button("🔄 Reset to Default"):
            st.session_state.weight_state = preset_options["Balanced (Default)"].copy()

        # Manual override sliders
        weights = {}
        for k in st.session_state.weight_state:
            weights[k] = st.sidebar.slider(f"{k.replace('_', ' ')} Weight (%)", 0, 100, st.session_state.weight_state[k], step=5)
            st.session_state.weight_state[k] = weights[k]

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            st.sidebar.error("Total weight cannot be zero. Please adjust sliders.")
            st.stop()

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Weight Pie Chart
        weight_df = pd.DataFrame({
            'Component': list(weights.keys()),
            'Weight': list(weights.values())
        })

        fig_weights = px.pie(weight_df, values='Weight', names='Component',
                             title='🧮 Current Weight Distribution',
                             hole=0.3)
        st.sidebar.plotly_chart(fig_weights, use_container_width=True)

    else:
        # Use fallback default weights (Balanced) when hidden
        weights = preset_options["Balanced (Default)"]
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # 3. Now compute Total_Score using normalized_weights
    df['Total_Score'] = (
        df['Score_Sales_Growth'] * normalized_weights['Sales_Growth'] +
        df['Score_Cost_Change'] * normalized_weights['Cost_Change'] +
        df['Score_GM_Change'] * normalized_weights['GM%_Change'] +
        #df['Score_Elasticity'] * normalized_weights['Elasticity'] +
        df['Score_GM_Abs_Change'] * normalized_weights['GM_Abs_Change']+
        df['Score_Qty_Change'] * normalized_weights['Qty_Change']+
        df['Score_ASP_Change'] * normalized_weights['ASP_Change']
    )



    st.sidebar.markdown("---")
    global_target = st.sidebar.slider("Global % Price Increase Target", 0.5, 10.0, 3.0, step=0.1)
    
    st.sidebar.markdown("### Inventory & Sales Coverage")

    total_months = st.sidebar.number_input("Total Months of Sales Data", min_value=1, max_value=36, value=6)
    stock_months = st.sidebar.number_input("Months Worth of Stock Available", min_value=0, max_value=12, value=0)
    impact_fraction = (total_months - stock_months) / total_months
    
    families = df['Product_Family'].dropna().unique().tolist()
    groups = df['Product_Group'].dropna().unique().tolist()
    selected_families = st.sidebar.multiselect("Override: Select Product Families", families)
    selected_groups = st.sidebar.multiselect("Override: Select Product Groups", groups)

    # Normalize score
    df['Score_Normalized'] = normalize_scores(df, 'Total_Score')
    df['Assigned_Price_Increase_%'] = np.nan

    # 1. Apply Product Group Overrides (Most granular)
    group_overrides = {grp: st.sidebar.slider(f"{grp} % Increase", 0.0, 20.0, 5.0, 0.5) for grp in selected_groups}
    df = apply_override(df, 'Product_Group', selected_groups, group_overrides, 'Score_Normalized', 'Price_Today', 'Revenue_1', 'ASP_1')

    # 2. Apply Product Family Overrides (only where Group not overridden)
    remaining_family_candidates = set(selected_families) - set(df[df['Assigned_Price_Increase_%'].notna()]['Product_Family'])
    family_overrides = {fam: st.sidebar.slider(f"{fam} % Increase", 0.0, 20.0, 5.0, 0.5) for fam in remaining_family_candidates}
    df = apply_override(df, 'Product_Family', family_overrides.keys(), family_overrides, 'Score_Normalized', 'Price_Today', 'Revenue_1', 'ASP_1')

    # 3. Apply Global Increase to remaining SKUs
    non_overridden_mask = df['Assigned_Price_Increase_%'].isna()
    df = apply_global_score_increase(df, non_overridden_mask, 'Score_Normalized', global_target)


    df['Estimated_Qty'] = df['Revenue_1'] / df['ASP_1']
    df['New_Price'] = df['Price_Today'] * (1 + df['Assigned_Price_Increase_%'] / 100)
    df['New_Revenue'] = df['Revenue_1'] * (1 + df['Assigned_Price_Increase_%'] / 100)

    #df['TTL_Cost'] = df['Revenue_1'] * (1 - df['GM%_1'] / 100)
    # Clean and convert relevant columns to numeric
    df['TTL_Cost'] = pd.to_numeric(df['TTL_Cost'], errors='coerce')
    df['Cost_Change_%'] = pd.to_numeric(df['Cost_Change_%'], errors='coerce')

    # Now safely perform the calculation
    df['Theoretical_New_Cost'] = df['TTL_Cost'] * (1 + df['Cost_Change_%'] / 100)

    # Optional: Fill NaNs if any
    df['Theoretical_New_Cost'] = df['Theoretical_New_Cost'].fillna(0)

    df['New_Cost'] = df['TTL_Cost'] + (df['Theoretical_New_Cost'] - df['TTL_Cost']) * impact_fraction
    #df.drop(columns=['Theoretical_New_Cost'], inplace=True)



    total_old_revenue = df['Revenue_1'].sum()
    target_total_revenue = total_old_revenue * (1 + global_target / 100)
    overridden_mask = df['Assigned_Price_Increase_%'].notna() & (
        df['Product_Family'].isin(family_overrides.keys()) | df['Product_Group'].isin(group_overrides.keys())
    )
    df = adjust_remaining(df, overridden_mask, target_total_revenue)

    # Final adjustment for total increase
    df['New_Revenue'] = df['Revenue_1'] * (1 + df['Assigned_Price_Increase_%'] / 100)
    df['Assigned_Price_Increase_%'] = (df['New_Price'] / df['Price_Today'] - 1) * 100

    full_summary = summarize_revenue(df,'Product_Family')

    total_row = pd.DataFrame({
        'Product_Family': ['TOTAL'],
        'Total_Revenue_Old': [df['Revenue_1'].sum()],
        'Total_Revenue_New': [df['New_Revenue'].sum()],
        'TTL_Cost': [df['TTL_Cost'].sum()],
        'New_Cost': [df['New_Cost'].sum()]
    })

    total_row['Revenue_Increase_%'] = (
        (total_row['Total_Revenue_New'] - total_row['Total_Revenue_Old']) / total_row['Total_Revenue_Old']
    ) * 100

    total_row['Cost_Increase_%'] = (
        (total_row['New_Cost'] - total_row['TTL_Cost']) / total_row['TTL_Cost']
    ) * 100

    total_row['Old_GM'] = total_row['Total_Revenue_Old'] - total_row['TTL_Cost']
    total_row['New_GM'] = total_row['Total_Revenue_New'] - total_row['New_Cost']
    total_row['GM_Impact'] = total_row['New_GM'] - total_row['Old_GM']


    full_summary = pd.concat([summarize_revenue(df,'Product_Family'), total_row], ignore_index=True)

    for col in ['Total_Revenue_Old', 'Total_Revenue_New', 'TTL_Cost', 'New_Cost', 'Old_GM', 'New_GM', 'GM_Impact']:
        full_summary[col] = full_summary[col].round(0).apply(lambda x: f"{x:,.0f}")

    for col in ['Revenue_Increase_%', 'Cost_Increase_%']:
        full_summary[col] = full_summary[col].round(2)


    full_summary['Revenue_Increase_%'] = full_summary['Revenue_Increase_%'].round(2)
    full_summary['Cost_Increase_%'] = full_summary['Cost_Increase_%'].round(2)

    st.subheader("📊 Summary of Price Impact")
    st.dataframe(full_summary, use_container_width=True)
    
    # Product Group Summary
    product_group_summary = summarize_revenue(df, 'Product_Group')

    # Format numbers
    for col in ['Total_Revenue_Old', 'Total_Revenue_New', 'TTL_Cost', 'New_Cost', 'Old_GM', 'New_GM', 'GM_Impact']:
        product_group_summary[col] = product_group_summary[col].round(0).apply(lambda x: f"{x:,.0f}")
    product_group_summary['Revenue_Increase_%'] = product_group_summary['Revenue_Increase_%'].round(2)
    product_group_summary['Cost_Increase_%'] = product_group_summary['Cost_Increase_%'].round(2)

    # Display it
    st.subheader("📘 Product Group Level Summary")
    st.dataframe(product_group_summary, use_container_width=True)
    
    #import plotly.express as px

    # --------------------------------------------
    # CLEAN & PREPARE full_summary FOR VISUALIZATION
    # --------------------------------------------
    viz_df = full_summary[full_summary['Product_Family'] != 'TOTAL'].copy()

    # Ensure revenue and GM columns are numeric
    cols_to_convert = ['Total_Revenue_Old', 'Total_Revenue_New', 'GM_Impact']
    for col in cols_to_convert:
        viz_df[col] = viz_df[col].replace(',', '', regex=True).astype(float)

    # Sort by old revenue
    viz_df = viz_df.sort_values(by='Total_Revenue_Old')

    # --------------------------------------------
    # 1️⃣ Revenue Before vs After (Grouped Bar)
    # --------------------------------------------
    import plotly.graph_objects as go

    # Melt revenue values
    melted = viz_df.melt(
        id_vars='Product_Family',
        value_vars=['Total_Revenue_Old', 'Total_Revenue_New'],
        var_name='Revenue Type',
        value_name='Amount'
    )

    # Create the figure
    fig1 = go.Figure()

    # Add grouped bar for revenue
    for revenue_type in melted['Revenue Type'].unique():
        data = melted[melted['Revenue Type'] == revenue_type]
        fig1.add_trace(go.Bar(
            x=data['Product_Family'],
            y=data['Amount'],
            name=revenue_type,
            yaxis='y1'
        ))

    # Add line for Assigned Price Increase %
    fig1.add_trace(go.Scatter(
        x=viz_df['Product_Family'],
        y=viz_df['Revenue_Increase_%'],
        name='Assigned Price Increase %',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='crimson', width=3, dash='dash')
    ))

    # Update layout for dual axis
    fig1.update_layout(
        title=' Revenue Before vs After Price Revision +  Assigned Price Increase %',
        xaxis=dict(title='Product Family'),
        yaxis=dict(
            title='Revenue (AED)',
            side='left'
        ),
        yaxis2=dict(
            title='Assigned Price Increase (%)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        barmode='group',
        xaxis_tickangle=-45,
        legend=dict(x=0.01, y=1.1, orientation='h'),
        margin=dict(t=60)
    )

    # Display in Streamlit
    st.plotly_chart(fig1, use_container_width=True)


    # --------------------------------------------
    # 2️⃣ Gross Margin Impact (Bar)
    # --------------------------------------------
    fig2 = px.bar(
        viz_df.sort_values(by='GM_Impact'),
        x='Product_Family',
        y='GM_Impact',
        title='📊 Gross Margin Impact by Product Family',
        labels={'GM_Impact': 'GM Impact (AED)', 'Product_Family': 'Product Family'},
        color='GM_Impact',
        color_continuous_scale='Blues'
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------
    # 3️⃣ Price Increase % Distribution (Histogram)
    # --------------------------------------------
    fig3 = px.histogram(
        df,
        x='Assigned_Price_Increase_%',
        nbins=20,
        title='📈 Distribution of Assigned Price Increase %',
        labels={'Assigned_Price_Increase_%': 'Price Increase (%)'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --------------------------------------------
    # 4️⃣ Score vs Price Increase (Scatter)
    # --------------------------------------------
    scatter_df = df[df['SKU'].notna()].copy()  # Ensure only SKU-level rows are plotted

    fig4 = px.scatter(
        scatter_df,
        x='Total_Score',
        y='Assigned_Price_Increase_%',
        size='Revenue_1',  # Size based on SKU revenue
        color='Product_Family',
        title='📈 Score vs Assigned Price Increase %',
        labels={
            'Total_Score': 'Composite Score',
            'Assigned_Price_Increase_%': 'Price Increase (%)',
            'Revenue_1': 'Revenue (AED)'
        },
        hover_data=['SKU', 'Product_Group', 'Price_Today', 'Revenue_1']
    )

    fig4.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color='DarkSlateGrey')))
    fig4.update_layout(xaxis_tickformat=".2f", yaxis_tickformat=".2f")

    st.plotly_chart(fig4, use_container_width=True)
    
     # 5️⃣ Revenue Curve vs Price Increase %
    # --------------------------------------------
    fig5 = px.scatter(
        df,
        x='Assigned_Price_Increase_%',
        y='New_Revenue',
        size='Revenue_1',
        color='Product_Family',
        title='📈 Revenue Curve by Price Increase %',
        labels={
            'Assigned_Price_Increase_%': 'Price Increase (%)',
            'New_Revenue': 'Estimated New Revenue',
            'Revenue_1': 'Base Revenue'
        },
        hover_data=['SKU', 'Total_Score', 'Price_Today']
    )
    fig5.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='gray')))
    st.plotly_chart(fig5, use_container_width=True)

    # ⬇️ Download CSV with all scoring logic
    csv_score_details = df[[
        'SKU', 'Sales_Growth_%', 'GM%_Change', 'Qty_Change_%','ASP_Change_%', 'Cost_Change_%', 'GM_Abs_Change',
        'Score_Sales_Growth', 'Score_GM_Change', 'Score_Qty_Change','Score_ASP_Change',
        'Score_Cost_Change', 'Score_GM_Abs_Change', 'Total_Score',
        'Assigned_Price_Increase_%', 'Price_Today', 'New_Price', 'Revenue_1', 'New_Revenue'
    ]].round(2).to_csv(index=False)

    st.download_button("📥 Download Detailed Scoring Sheet", data=csv_score_details,
                       file_name="SKU_Score_Details.csv")


    csv = df[['SKU', 'Product_Family', 'Price_Today', 'Assigned_Price_Increase_%',
              'New_Price', 'Revenue_1', 'New_Revenue']].round(2).to_csv(index=False)
    st.download_button("📥 Download SKU-Level Price Plan", data=csv, file_name="price_revision_output.csv")
else:
    st.warning("⚠️ Please upload all six input files to start.")
