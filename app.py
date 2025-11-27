import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# =========================================================
# STREAMLIT PAGE CONFIG & GLOBAL SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI-Driven Smart Business Opportunity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use a nice dark theme for Plotly
px.defaults.template = "plotly_dark"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

EXPECTED_COLUMNS = [
    "Opportunity ID",
    "Opportunity Name",
    "Customer Name",
    "Opportunity Source",
    "Domain",
    "Solution Type",
    "Vendor(s) Involved",
    "Opportunity Stage",
    "Solution Size Type",
    "Expected Value (SAR)",
    "Expected Closing Date ( Deadline)",
    "Proposal Submission Date ( TP sent to BDO)",
    "Presales Owner",
    "Sales Owner",
    "Technical Status",
    "Opportunity Status",
    "Last Updated Date",
    "Win/Loss Reason",
    "Final Contract Value (SAR)",
]

REGULATORY_KEYWORDS = {
    "NCA": [
        "nca",
        "national cybersecurity",
        "ecc",
        "essential cyber",
        "cst",
        "cybersecurity framework",
        "national cyber",
    ],
    "SAMA": [
        "sama",
        "bank",
        "core banking",
        "payment",
        "pos",
        "cards",
        "digital banking",
        "finance",
        "financial",
    ],
    "CMA": [
        "cma",
        "capital market",
        "brokerage",
        "trading",
        "investment",
        "asset management",
        "securities",
    ],
}


def parse_date(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def map_status(status):
    if pd.isna(status):
        return np.nan
    s = str(status).strip().lower()
    if "won" in s or "success" in s:
        return "Won"
    elif "lost" in s or "closed lost" in s or "cancel" in s:
        return "Lost"
    else:
        return "In Progress"


def tag_regulatory_drivers(row):
    text_cols = [
        "Opportunity Name",
        "Domain",
        "Solution Type",
        "Win/Loss Reason",
    ]
    combined = " ".join(str(row.get(c, "")) for c in text_cols).lower()
    drivers = []
    for reg, kws in REGULATORY_KEYWORDS.items():
        if any(kw in combined for kw in kws):
            drivers.append(reg)
    if not drivers:
        drivers.append("General")
    return drivers


def fmt_number(x):
    """Format number with commas and 2 decimals for display."""
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return x


def format_currency_columns(df, columns):
    """Return a copy of df with formatted currency columns (for display only)."""
    df_disp = df.copy()
    for c in columns:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].apply(lambda v: fmt_number(v) if pd.notna(v) else "")
    return df_disp


def categorize_gap(reason):
    """High-level category for each gap reason."""
    if pd.isna(reason) or not str(reason).strip():
        return "No Gap"

    text = str(reason).lower()

    if "high-value opportunity close to deadline" in text:
        return "Deadline Risk (High Value)"
    if "no proposal submission date" in text:
        return "Missing Proposal"
    if "not updated for more than 30 days" in text:
        return "Stale Opportunity"
    if "deviation between expected and final contract value" in text:
        return "Value Mismatch"
    if "deadline reached/passed" in text:
        return "Deadline Passed"
    return "Other"


def detect_business_gaps(df, today=None):
    if today is None:
        today = pd.Timestamp.today()

    data = df.copy()
    reasons = []

    for _, row in data.iterrows():
        row_reasons = []

        # Rule 1: High value, near deadline, still in progress
        ev = row.get("Expected Value (SAR)", np.nan)
        try:
            high_value = pd.notna(ev) and ev > 100000  # adjustable threshold
        except Exception:
            high_value = False

        days_to_deadline = row.get("days_to_deadline", np.nan)
        status = row.get("Status_Simplified", "")

        if (
            high_value
            and pd.notna(days_to_deadline)
            and days_to_deadline < 7
            and status == "In Progress"
        ):
            row_reasons.append(
                "High-value opportunity close to deadline still in progress."
            )

        # Rule 2: No proposal but deadline reached/passed
        proposal_date = row.get("Proposal Submission Date ( TP sent to BDO)")
        if pd.isna(proposal_date) and pd.notna(days_to_deadline) and days_to_deadline <= 0:
            row_reasons.append(
                "Deadline reached/passed but no proposal submission date recorded."
            )

        # Rule 3: Not updated for long time
        days_since_update = row.get("days_since_last_update", np.nan)
        if (
            pd.notna(days_since_update)
            and days_since_update > 30
            and status == "In Progress"
        ):
            row_reasons.append(
                "Opportunity not updated for more than 30 days but still in progress."
            )

        # Rule 4: Large deviation between expected and final value
        expected_val = row.get("Expected Value (SAR)", np.nan)
        final_val = row.get("Final Contract Value (SAR)", np.nan)
        if pd.notna(expected_val) and pd.notna(final_val) and status in ["Won", "Lost"]:
            diff_ratio = abs(final_val - expected_val) / max(expected_val, 1)
            if diff_ratio > 0.3:
                row_reasons.append(
                    "Significant deviation between expected and final contract value."
                )

        reasons.append("; ".join(row_reasons) if row_reasons else "")

    data["Gap_Reason"] = reasons
    data["Gap_Flag"] = data["Gap_Reason"].apply(lambda x: 1 if x else 0)
    data["Gap_Category"] = data["Gap_Reason"].apply(categorize_gap)
    return data


def build_customer_kpis(df):
    if "Customer Name" not in df.columns:
        return pd.DataFrame()

    records = []
    grouped = df.groupby("Customer Name")

    for cust, grp in grouped:
        total_opps = len(grp)

        won_mask = grp["Status_Simplified"] == "Won"
        lost_mask = grp["Status_Simplified"] == "Lost"
        inprog_mask = grp["Status_Simplified"] == "In Progress"

        won_count = int(won_mask.sum())
        lost_count = int(lost_mask.sum())
        inprog_count = int(inprog_mask.sum())

        closed = won_count + lost_count
        win_rate = won_count / closed if closed > 0 else np.nan

        total_revenue = grp.loc[won_mask, "Final Contract Value (SAR)"].sum()

        recency = grp["days_since_last_update"].min()
        frequency = total_opps
        monetary = total_revenue

        if pd.notna(recency) and recency >= 0:
            recency_component = 1 / (1 + recency)
        else:
            recency_component = 0.0

        engagement_score = (
            recency_component
            + np.log1p(frequency)
            + np.log1p(monetary / 100000.0 + 1e-9)
        )

        pipeline = grp[inprog_mask].copy()
        if not pipeline.empty:
            pipeline_expected_value = (
                pipeline["Predicted_Final_Value"]
                .fillna(pipeline["Expected Value (SAR)"])
                * pipeline["Predicted_Win_Prob"].fillna(0.5)
            ).sum()
        else:
            pipeline_expected_value = 0.0

        reg_counts = (
            grp.explode("Regulatory_Drivers")["Regulatory_Drivers"]
            .value_counts()
            .to_dict()
        )

        records.append(
            {
                "Customer Name": cust,
                "Total_Opportunities": total_opps,
                "Won_Count": won_count,
                "Lost_Count": lost_count,
                "InProgress_Count": inprog_count,
                "Win_Rate": win_rate,
                "Historical_Revenue_SAR": total_revenue,
                "Engagement_Score": engagement_score,
                "Pipeline_Expected_Revenue_SAR": pipeline_expected_value,
                "Regulatory_Driver_Counts": reg_counts,
            }
        )

    return pd.DataFrame(records)


def explain_customer_win_loss(customer_name, df):
    cust_data = df[df["Customer Name"] == customer_name].copy()
    if cust_data.empty:
        return {}

    won = cust_data[cust_data["Status_Simplified"] == "Won"]
    lost = cust_data[cust_data["Status_Simplified"] == "Lost"]

    won_count = len(won)
    lost_count = len(lost)
    closed = won_count + lost_count
    win_rate = won_count / closed if closed > 0 else np.nan

    def top_counts(df_in, col, n=5):
        if col not in df_in.columns or df_in.empty:
            return {}
        return df_in[col].value_counts().head(n).to_dict()

    summary = {
        "customer_name": customer_name,
        "total_opportunities": len(cust_data),
        "won_count": won_count,
        "lost_count": lost_count,
        "win_rate": win_rate,
        "historical_revenue_sar": won["Final Contract Value (SAR)"].sum(),
        "dominant_domains_won": top_counts(won, "Domain"),
        "dominant_domains_lost": top_counts(lost, "Domain"),
        "dominant_solutions_won": top_counts(won, "Solution Type"),
        "dominant_solutions_lost": top_counts(lost, "Solution Type"),
        "dominant_stage_won": top_counts(won, "Opportunity Stage"),
        "dominant_stage_lost": top_counts(lost, "Opportunity Stage"),
    }

    def reg_summary(df_in):
        if df_in.empty:
            return {}
        exploded = df_in.explode("Regulatory_Drivers")
        return exploded["Regulatory_Drivers"].value_counts().to_dict()

    summary["regulatory_won"] = reg_summary(won)
    summary["regulatory_lost"] = reg_summary(lost)

    insights = []

    if not np.isnan(win_rate):
        if win_rate > 0.7:
            insights.append(
                "You have a strong win rate with this customer; relationship and positioning are very positive."
            )
        elif win_rate < 0.4:
            insights.append(
                "Win rate with this customer is relatively low; review qualification, pricing, and solution fit."
            )
        else:
            insights.append(
                "Win rate with this customer is moderate; there is room to grow share of wallet."
            )

    if summary["dominant_domains_won"]:
        top_win_domain = next(iter(summary["dominant_domains_won"].keys()))
        insights.append(
            f"You most frequently win in the '{top_win_domain}' domain with this customer."
        )
    if summary["dominant_domains_lost"]:
        top_loss_domain = next(iter(summary["dominant_domains_lost"].keys()))
        if (
            summary["dominant_domains_won"]
            and top_loss_domain not in summary["dominant_domains_won"]
        ):
            insights.append(
                f"Losses often occur in the '{top_loss_domain}' domain; consider strengthening your portfolio or partner ecosystem here."
            )

    if summary["regulatory_won"]:
        top_reg_win = max(
            summary["regulatory_won"], key=summary["regulatory_won"].get
        )
        if top_reg_win != "General":
            insights.append(
                f"Wins with this customer are strongly associated with {top_reg_win}-driven projects."
            )
    if summary["regulatory_lost"]:
        top_reg_loss = max(
            summary["regulatory_lost"], key=summary["regulatory_lost"].get
        )
        if top_reg_loss != "General":
            insights.append(
                f"Many losses are linked to {top_reg_loss}-driven projects; review how well your solutions align with that regulator’s controls."
            )

    if summary["dominant_stage_lost"]:
        top_loss_stage = next(iter(summary["dominant_stage_lost"].keys()))
        insights.append(
            f"Most losses happen around the '{top_loss_stage}' stage; investigate what typically goes wrong at this point."
        )

    summary["insights"] = insights
    return summary


def regulatory_business_summary(df):
    records = []
    for cust, grp in df.groupby("Customer Name"):
        won = grp[grp["Status_Simplified"] == "Won"].copy()
        if not won.empty:
            won_exploded = won.explode("Regulatory_Drivers")
            hist_by_reg = won_exploded.groupby("Regulatory_Drivers")[
                "Final Contract Value (SAR)"
            ].sum()
        else:
            hist_by_reg = pd.Series(dtype=float)

        inprog = grp[grp["Status_Simplified"] == "In Progress"].copy()
        if not inprog.empty:
            inprog_exploded = inprog.explode("Regulatory_Drivers")
            inprog_exploded["Expected_Pipeline_Value"] = (
                inprog_exploded["Predicted_Final_Value"]
                .fillna(inprog_exploded["Expected Value (SAR)"])
                * inprog_exploded["Predicted_Win_Prob"].fillna(0.5)
            )
            pipe_by_reg = inprog_exploded.groupby("Regulatory_Drivers")[
                "Expected_Pipeline_Value"
            ].sum()
        else:
            pipe_by_reg = pd.Series(dtype=float)

        all_regs = set(hist_by_reg.index).union(set(pipe_by_reg.index))
        for reg in all_regs:
            records.append(
                {
                    "Customer Name": cust,
                    "Regulatory_Driver": reg,
                    "Historical_Revenue_SAR": float(hist_by_reg.get(reg, 0.0)),
                    "Expected_Pipeline_Revenue_SAR": float(
                        pipe_by_reg.get(reg, 0.0)
                    ),
                }
            )

    return pd.DataFrame(records)


# =========================================================
# MAIN ANALYSIS PIPELINE (CACHED)
# =========================================================

@st.cache_data(show_spinner=True)
def run_full_analysis(uploaded_bytes, filename):
    # Load file
    ext = filename.split(".")[-1].lower()
    if ext in ["xlsx", "xls"]:
        df_raw = pd.read_excel(uploaded_bytes)
    else:
        df_raw = pd.read_csv(uploaded_bytes)

    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Missing expected columns (for info only)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    # Parse dates
    date_cols = [
        "Expected Closing Date ( Deadline)",
        "Proposal Submission Date ( TP sent to BDO)",
        "Last Updated Date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = parse_date(df[col])

    # Numeric columns
    numeric_cols = ["Expected Value (SAR)", "Final Contract Value (SAR)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("SAR", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering (time deltas)
    today = pd.Timestamp.today()

    if "Expected Closing Date ( Deadline)" in df.columns:
        df["days_to_deadline"] = (
            df["Expected Closing Date ( Deadline)"] - today
        ).dt.days
    if "Proposal Submission Date ( TP sent to BDO)" in df.columns:
        df["days_since_proposal"] = (
            today - df["Proposal Submission Date ( TP sent to BDO)"]
        ).dt.days
    if "Last Updated Date" in df.columns:
        df["days_since_last_update"] = (
            today - df["Last Updated Date"]
        ).dt.days

    # Simplified status
    if "Opportunity Status" in df.columns:
        df["Status_Simplified"] = df["Opportunity Status"].apply(map_status)
    else:
        df["Status_Simplified"] = np.nan

    # --------------------------
    # Classification model (predict Won/Lost/In Progress)
    # --------------------------
    classification_features = [
        "Opportunity Source",
        "Domain",
        "Solution Type",
        "Vendor(s) Involved",
        "Opportunity Stage",
        "Solution Size Type",
        "Expected Value (SAR)",
        "days_to_deadline",
        "days_since_proposal",
        "days_since_last_update",
    ]
    cls_df = df.dropna(subset=["Status_Simplified"]).copy()

    df["Predicted_Status"] = np.nan
    df["Predicted_Win_Prob"] = np.nan

    if not cls_df.empty and cls_df["Status_Simplified"].nunique() >= 2:
        X_cls = cls_df[classification_features]
        y_cls = cls_df["Status_Simplified"]

        cat_features = [
            c for c in classification_features
            if c in cls_df.columns and cls_df[c].dtype == "object"
        ]
        num_features = [
            c for c in classification_features
            if c in cls_df.columns and c not in cat_features
        ]

        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        num_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_features),
                ("num", num_transformer, num_features),
            ]
        )

        cls_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        cls_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", cls_model)]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )
        cls_pipeline.fit(X_train, y_train)

        # Predict for whole dataset
        mask_valid = df[classification_features].notnull().any(axis=1)
        X_all = df.loc[mask_valid, classification_features]
        preds = cls_pipeline.predict(X_all)
        proba = cls_pipeline.predict_proba(X_all)
        classes = cls_pipeline.classes_
        win_index = list(classes).index("Won") if "Won" in classes else None

        df.loc[mask_valid, "Predicted_Status"] = preds
        if win_index is not None:
            df.loc[mask_valid, "Predicted_Win_Prob"] = proba[:, win_index]

        # Metrics
        y_pred_test = cls_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    else:
        accuracy = np.nan
        f1 = np.nan

    # --------------------------
    # Regression model (predict final value)
    # --------------------------
    df["Predicted_Final_Value"] = np.nan
    if "Final Contract Value (SAR)" in df.columns:
        reg_df = df.dropna(subset=["Final Contract Value (SAR)"]).copy()
        regression_features = classification_features

        if not reg_df.empty:
            X_reg = reg_df[regression_features]
            y_reg = reg_df["Final Contract Value (SAR)"]

            cat_features_reg = [
                c for c in regression_features
                if c in reg_df.columns and reg_df[c].dtype == "object"
            ]
            num_features_reg = [
                c for c in regression_features
                if c in reg_df.columns and c not in cat_features_reg
            ]

            cat_transformer_reg = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            num_transformer_reg = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median"))]
            )

            preprocessor_reg = ColumnTransformer(
                transformers=[
                    ("cat", cat_transformer_reg, cat_features_reg),
                    ("num", num_transformer_reg, num_features_reg),
                ]
            )

            reg_model = RandomForestRegressor(
                n_estimators=300, random_state=42
            )
            reg_pipeline = Pipeline(
                steps=[("preprocessor", preprocessor_reg), ("model", reg_model)]
            )

            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )
            reg_pipeline.fit(X_train_r, y_train_r)

            mask_valid_r = df[regression_features].notnull().any(axis=1)
            X_all_reg = df.loc[mask_valid_r, regression_features]
            preds_reg = reg_pipeline.predict(X_all_reg)
            df.loc[mask_valid_r, "Predicted_Final_Value"] = preds_reg

            y_pred_r = reg_pipeline.predict(X_test_r)
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
            r2 = r2_score(y_test_r, y_pred_r)
        else:
            rmse = np.nan
            r2 = np.nan
    else:
        rmse = np.nan
        r2 = np.nan

    # --------------------------
    # Clustering
    # --------------------------
    cluster_features = [
        "Expected Value (SAR)",
        "Final Contract Value (SAR)",
        "days_to_deadline",
        "days_since_proposal",
        "days_since_last_update",
    ]
    df["Opportunity_Cluster"] = np.nan
    cluster_df = df[cluster_features].dropna()

    if len(cluster_df) >= 5:
        scaler = StandardScaler()
        cluster_scaled = scaler.fit_transform(cluster_df)
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_scaled)
        df.loc[cluster_df.index, "Opportunity_Cluster"] = labels

    # Regulatory tagging
    df["Regulatory_Drivers"] = df.apply(tag_regulatory_drivers, axis=1)
    df["Primary_Regulatory_Driver"] = df["Regulatory_Drivers"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "General"
    )

    # Business gaps & categories
    df = detect_business_gaps(df, today=today)

    # Customer KPIs & regulatory summary
    customer_kpis = build_customer_kpis(df)
    reg_business = regulatory_business_summary(df)

    metrics = {
        "classification_accuracy": accuracy,
        "classification_f1": f1,
        "regression_rmse": rmse,
        "regression_r2": r2,
        "missing_columns": missing_cols,
    }

    return df, customer_kpis, reg_business, metrics


# =========================================================
# FILE UPLOAD & MAIN UI
# =========================================================

st.title("AI-Driven Smart Business Opportunity Analyzer")
st.write(
    "Upload your **opportunity tracking sheet** (Excel/CSV). "
    "The app will analyse opportunities, train ML models, and generate insights per customer and regulator."
)

uploaded_file = st.file_uploader(
    "Upload Opportunity Excel/CSV", type=["xlsx", "xls", "csv"]
)

if not uploaded_file:
    st.info(
        "Please upload your **opportunity tracking sheet**. "
        "No data is stored outside this running app."
    )
    st.stop()

# Run analysis
with st.spinner("Running AI analysis on your opportunities..."):
    df_pred, customer_kpis, reg_business, metrics = run_full_analysis(
        uploaded_file.getvalue(), uploaded_file.name
    )

st.success("Analysis complete.")

# =========================================================
# HIGH-LEVEL METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)
total_opps = len(df_pred)
won = (df_pred["Status_Simplified"] == "Won").sum()
lost = (df_pred["Status_Simplified"] == "Lost").sum()
inprog = (df_pred["Status_Simplified"] == "In Progress").sum()

with col1:
    st.metric("Total Opportunities", total_opps)
with col2:
    st.metric("Won", int(won))
with col3:
    st.metric("Lost", int(lost))
with col4:
    st.metric("In Progress", int(inprog))

if metrics["classification_accuracy"] == metrics["classification_accuracy"]:  # not NaN
    st.caption(
        f"Classification Accuracy: **{metrics['classification_accuracy']:.3f}**, "
        f"F1 (macro): **{metrics['classification_f1']:.3f}**"
    )
if metrics["regression_r2"] == metrics["regression_r2"]:
    st.caption(
        f"Regression RMSE: **{metrics['regression_rmse']:.2f}**, "
        f"R²: **{metrics['regression_r2']:.3f}**"
    )
if metrics["missing_columns"]:
    st.warning(
        f"Missing expected columns in uploaded file: {metrics['missing_columns']}"
    )

# =========================================================
# TABS FOR DETAILED VIEWS
# =========================================================

tab_overview, tab_opps, tab_customers, tab_reg = st.tabs(
    ["📊 Data Overview", "🎯 Opportunities AI", "🤝 Customer Insights", "⚖ Regulatory View"]
)

# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab_overview:
    st.subheader("Business Overview Dashboard")

    # ---------- Status & Value by Status ----------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Opportunities by Status (Count)**")

        status_counts = (
            df_pred["Status_Simplified"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )

        fig_status_count = px.bar(
            status_counts,
            x="Status",
            y="Count",
            labels={"Status": "Status", "Count": "Number of opportunities"},
            text="Count",
        )
        fig_status_count.update_traces(textposition="outside")
        st.plotly_chart(fig_status_count, use_container_width=True)

    with colB:
        if "Expected Value (SAR)" in df_pred.columns:
            st.markdown("**Total Expected Value by Status (SAR)**")
            status_value = (
                df_pred.groupby("Status_Simplified")["Expected Value (SAR)"]
                .sum()
                .reset_index()
            )
            fig_status_value = px.bar(
                status_value,
                x="Status_Simplified",
                y="Expected Value (SAR)",
                labels={
                    "Status_Simplified": "Status",
                    "Expected Value (SAR)": "Total expected value (SAR)",
                },
                text="Expected Value (SAR)",
            )
            fig_status_value.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside"
            )
            st.plotly_chart(fig_status_value, use_container_width=True)

    if "Expected Value (SAR)" in df_pred.columns:
        total_expected = df_pred["Expected Value (SAR)"].sum()
        median_expected = df_pred["Expected Value (SAR)"].median()
        st.markdown(
            f"- **Total expected pipeline**: **{fmt_number(total_expected)} SAR**  \n"
            f"- **Median deal size**: **{fmt_number(median_expected)} SAR**  \n"
            f"- The charts above show **how many deals** you have per status and "
            f"**where most of your expected revenue is concentrated**."
        )

    st.markdown("---")

    # ---------- Value Distributions ----------
    st.markdown("### Opportunity Value Distributions")
    col1, col2 = st.columns(2)

    with col1:
        if "Expected Value (SAR)" in df_pred.columns:
            st.markdown("**Expected Value (SAR)**")
            fig_ev = px.histogram(
                df_pred,
                x="Expected Value (SAR)",
                nbins=30,
                labels={
                    "Expected Value (SAR)": "Expected value range (SAR)",
                    "count": "Number of opportunities",
                },
            )
            st.plotly_chart(fig_ev, use_container_width=True)
            st.caption(
                "Each bar shows how many opportunities fall into a given value range. "
                "Tall bars on the left = many small deals; tall bars on the right = more large deals."
            )

    with col2:
        if "Final Contract Value (SAR)" in df_pred.columns:
            st.markdown("**Final Contract Value (SAR)**")
            fig_fv = px.histogram(
                df_pred,
                x="Final Contract Value (SAR)",
                nbins=30,
                labels={
                    "Final Contract Value (SAR)": "Final contract value range (SAR)",
                    "count": "Number of closed deals",
                },
            )
            st.plotly_chart(fig_fv, use_container_width=True)
            st.caption(
                "This shows the actual sizes of closed deals. Comparing this with the expected value "
                "helps you see if estimates are usually too high or too low."
            )

    st.markdown("---")

    # ---------- Top Customers by Expected Pipeline ----------
    st.markdown("### Top Customers by Expected Pipeline Revenue")

    if not customer_kpis.empty:
        top_cust = customer_kpis.sort_values(
            by="Pipeline_Expected_Revenue_SAR", ascending=False
        ).head(10)

        fig_cust = px.bar(
            top_cust,
            x="Customer Name",
            y="Pipeline_Expected_Revenue_SAR",
            labels={
                "Customer Name": "Customer",
                "Pipeline_Expected_Revenue_SAR": "Expected pipeline revenue (SAR)",
            },
            text="Pipeline_Expected_Revenue_SAR",
        )
        fig_cust.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside"
        )
        st.plotly_chart(fig_cust, use_container_width=True)
        st.caption(
            "These are the customers where the **future opportunity pipeline is strongest** "
            "based on predicted values and win probabilities."
        )
    else:
        st.info("Customer KPIs were not computed (no customer data).")

    st.markdown("---")

    # ---------- Gaps Detected (PROFESSIONAL VISUALS) ----------
    st.markdown("### Opportunities with Detected Business Gaps")

    gap_df = df_pred[df_pred["Gap_Flag"] == 1].copy()
    gap_count = len(gap_df)
    st.write(f"Number of opportunities with detected gaps: **{int(gap_count)}**")

    if gap_count > 0:
        # Ensure Gap_Category exists (safety)
        if "Gap_Category" not in gap_df.columns:
            gap_df["Gap_Category"] = gap_df["Gap_Reason"].apply(categorize_gap)

        # 1) Bar chart: number of gaps by category
        gap_counts = (
            gap_df["Gap_Category"]
            .value_counts()
            .rename_axis("Gap_Category")
            .reset_index(name="Count")
        )

        # 2) Value impact by gap category (expected value)
        if "Expected Value (SAR)" in gap_df.columns:
            gap_value = (
                gap_df.groupby("Gap_Category")["Expected Value (SAR)"]
                .sum()
                .reset_index()
            )
        else:
            gap_value = pd.DataFrame(columns=["Gap_Category", "Expected Value (SAR)"])

        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Gap frequency by category**")
            fig_gap_count = px.bar(
                gap_counts,
                x="Gap_Category",
                y="Count",
                text="Count",
                labels={"Gap_Category": "Gap Category", "Count": "Number of opportunities"},
            )
            fig_gap_count.update_traces(textposition="outside")
            fig_gap_count.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_gap_count, use_container_width=True)

        with g2:
            st.markdown("**Total expected value at risk by gap category (SAR)**")
            if not gap_value.empty:
                fig_gap_val = px.bar(
                    gap_value,
                    x="Gap_Category",
                    y="Expected Value (SAR)",
                    text="Expected Value (SAR)",
                    labels={
                        "Gap_Category": "Gap Category",
                        "Expected Value (SAR)": "Expected value at risk (SAR)",
                    },
                )
                fig_gap_val.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig_gap_val.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig_gap_val, use_container_width=True)
            else:
                st.info("No expected value data available for gaps.")

        # 3) Heatmap: Gap category vs Opportunity Stage
        st.markdown("**Gap heatmap by opportunity stage**")
        if "Opportunity Stage" in gap_df.columns:
            heat_data = (
                gap_df.groupby(["Gap_Category", "Opportunity Stage"])
                .size()
                .reset_index(name="Count")
            )
            fig_heat = px.density_heatmap(
                heat_data,
                x="Opportunity Stage",
                y="Gap_Category",
                z="Count",
                color_continuous_scale="Reds",
            )
            fig_heat.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No 'Opportunity Stage' column available for gap heatmap.")

        # Detailed table (formatted money)
        money_cols = ["Expected Value (SAR)"]
        gap_display = format_currency_columns(
            gap_df[
                [
                    "Opportunity ID",
                    "Opportunity Name",
                    "Customer Name",
                    "Status_Simplified",
                    "Expected Value (SAR)",
                    "days_to_deadline",
                    "days_since_last_update",
                    "Gap_Category",
                    "Gap_Reason",
                ]
            ],
            money_cols,
        )

        with st.expander("Show opportunities with gaps (detailed table)"):
            st.dataframe(gap_display)

    else:
        st.info("No gaps detected based on current rules.")

    with st.expander("Show raw dataset (first 200 rows)"):
        money_cols = ["Expected Value (SAR)", "Final Contract Value (SAR)"]
        st.dataframe(format_currency_columns(df_pred.head(200), money_cols))


# =========================================================
# TAB 2: OPPORTUNITIES AI
# =========================================================

with tab_opps:
    st.subheader("Opportunity-Level Predictions and Insights")

    customer_filter = st.selectbox(
        "Filter by customer (optional)",
        options=["All"] + sorted(df_pred["Customer Name"].dropna().unique().tolist()),
    )

    filtered = df_pred.copy()
    if customer_filter != "All":
        filtered = filtered[filtered["Customer Name"] == customer_filter]

    cols_to_show = [
        "Opportunity ID",
        "Opportunity Name",
        "Customer Name",
        "Opportunity Stage",
        "Status_Simplified",
        "Predicted_Status",
        "Predicted_Win_Prob",
        "Expected Value (SAR)",
        "Final Contract Value (SAR)",
        "Predicted_Final_Value",
        "Primary_Regulatory_Driver",
        "Opportunity_Cluster",
        "Gap_Flag",
        "Gap_Category",
        "Gap_Reason",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    money_cols = ["Expected Value (SAR)", "Final Contract Value (SAR)", "Predicted_Final_Value"]
    filtered_display = format_currency_columns(filtered[cols_to_show], money_cols)

    st.dataframe(
        filtered_display.sort_values(
            by="Predicted_Win_Prob", ascending=False
        )
    )

    st.download_button(
        "Download opportunities with predictions (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="opportunity_predictions.csv",
        mime="text/csv",
    )


# =========================================================
# TAB 3: CUSTOMER INSIGHTS
# =========================================================

with tab_customers:
    st.subheader("Per-Customer Analytics & Explanations")

    if customer_kpis.empty:
        st.info("Customer KPIs could not be computed.")
    else:
        customer_kpis_sorted = customer_kpis.sort_values(
            by=["Pipeline_Expected_Revenue_SAR", "Historical_Revenue_SAR"],
            ascending=False,
        )

        st.subheader("Top Customers by Expected Pipeline Revenue")
        cust_display = customer_kpis_sorted[
            [
                "Customer Name",
                "Win_Rate",
                "Historical_Revenue_SAR",
                "Pipeline_Expected_Revenue_SAR",
                "Engagement_Score",
            ]
        ].copy()
        cust_display = format_currency_columns(
            cust_display,
            ["Historical_Revenue_SAR", "Pipeline_Expected_Revenue_SAR"],
        )
        st.dataframe(cust_display.head(20))

        cust_names = sorted(customer_kpis["Customer Name"].unique().tolist())
        selected_customer = st.selectbox("Select a customer", options=cust_names)

        cust_summary = explain_customer_win_loss(selected_customer, df_pred)
        if cust_summary:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Opps", cust_summary["total_opportunities"])
            with c2:
                st.metric("Won", cust_summary["won_count"])
            with c3:
                st.metric("Lost", cust_summary["lost_count"])
            with c4:
                st.metric(
                    "Win Rate",
                    f"{cust_summary['win_rate']*100:.1f}%"
                    if cust_summary["win_rate"] == cust_summary["win_rate"]
                    else "N/A",
                )

            st.markdown("**Insights:**")
            for ins in cust_summary["insights"]:
                st.write(f"- {ins}")

            st.markdown("**This customer's opportunities:**")
            cust_opps = df_pred[df_pred["Customer Name"] == selected_customer][
                [
                    "Opportunity ID",
                    "Opportunity Name",
                    "Opportunity Stage",
                    "Status_Simplified",
                    "Predicted_Status",
                    "Predicted_Win_Prob",
                    "Expected Value (SAR)",
                    "Final Contract Value (SAR)",
                    "Predicted_Final_Value",
                    "Primary_Regulatory_Driver",
                    "Gap_Flag",
                    "Gap_Category",
                    "Gap_Reason",
                ]
            ]
            cust_opps_display = format_currency_columns(
                cust_opps,
                ["Expected Value (SAR)", "Final Contract Value (SAR)", "Predicted_Final_Value"],
            )
            st.dataframe(cust_opps_display)


# =========================================================
# TAB 4: REGULATORY VIEW
# =========================================================

with tab_reg:
    st.subheader("Business Driven by NCA / SAMA / CMA / General")

    if reg_business.empty:
        st.info("Regulatory business summary is empty.")
    else:
        reg_filter = st.selectbox(
            "Select regulatory framework",
            options=sorted(reg_business["Regulatory_Driver"].unique().tolist()),
        )
        reg_filtered = reg_business[
            reg_business["Regulatory_Driver"] == reg_filter
        ]

        reg_display = reg_filtered.copy()
        reg_display = format_currency_columns(
            reg_display,
            ["Historical_Revenue_SAR", "Expected_Pipeline_Revenue_SAR"],
        )

        st.write(f"**{reg_filter} - Historical and Expected Revenue by Customer**")
        st.dataframe(
            reg_display.sort_values(
                by="Expected_Pipeline_Revenue_SAR", ascending=False
            ).head(50)
        )

        fig_reg = px.bar(
            reg_filtered.sort_values(
                by="Expected_Pipeline_Revenue_SAR", ascending=False
            ).head(20),
            x="Customer Name",
            y=["Historical_Revenue_SAR", "Expected_Pipeline_Revenue_SAR"],
            barmode="group",
            title=f"{reg_filter} - Top Customers by Historical & Expected Revenue",
        )
        st.plotly_chart(fig_reg, use_container_width=True)
