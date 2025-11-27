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

# Use a light / corporate template for Plotly
px.defaults.template = "plotly_white"

# Simple CSS for a light blue-accent corporate theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #12355b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPER FUNCTIONS & KNOWLEDGE BASE
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

# Mapping of sector -> primary regulator
SECTOR_REGULATOR_MAP = {
    "Banking": "SAMA",
    "Fintech": "SAMA",
    "Payments": "SAMA",
    "Insurance": "SAMA",
    "Capital Markets": "CMA",
    "Investment": "CMA",
    "Government": "NCA",
    "Critical Infrastructure": "NCA",
    "Healthcare": "NCA",
    "Education": "General",
    "Private": "General",
    "Other": "General",
}

# ---- FULL CONTROL LIBRARY (GROUPED BY CATEGORY) ----
CONTROL_LIBRARY = {
    "Governance & Strategy": [
        "Cybersecurity Strategy",
        "Cybersecurity Governance Framework",
        "Cybersecurity Policy Management",
        "Cybersecurity Roles & Responsibilities",
        "Cybersecurity Awareness & Training Program",
        "Third-Party Cybersecurity Governance",
    ],
    "Risk Management": [
        "Cybersecurity Risk Assessment Process",
        "Risk Register & Treatment Plan",
        "Vendor / Third-Party Risk Management",
        "Business Impact Analysis (BIA)",
        "Risk-Based Prioritization of Controls",
    ],
    "Asset & Configuration Management": [
        "IT Asset Inventory",
        "OT/ICS Asset Inventory",
        "Configuration Baselines",
        "Secure Configuration Management",
        "Patch Management Process",
        "Software & Hardware Whitelisting",
    ],
    "Identity & Access Management": [
        "Identity Lifecycle Management",
        "Onboarding & Offboarding Process",
        "Role-Based Access Control (RBAC)",
        "Multi-Factor Authentication (MFA)",
        "Privileged Access Management (PAM)",
        "Directory Services & Identity Store (AD/LDAP)",
    ],
    "Network Security": [
        "Network Segmentation Design",
        "Next-Generation Firewall (NGFW)",
        "Web Application Firewall (WAF)",
        "Intrusion Detection / Intrusion Prevention (IDS/IPS)",
        "Network Access Control (NAC)",
        "Secure Remote Access (VPN / ZTNA)",
    ],
    "Endpoint & Server Security": [
        "Endpoint Protection Platform (EPP)",
        "Endpoint Detection & Response (EDR)",
        "Extended Detection & Response (XDR)",
        "Server Hardening",
        "Mobile Device Management (MDM/UEM)",
        "Application Control / Whitelisting on Endpoints",
    ],
    "Data Protection & Privacy": [
        "Data Classification Framework",
        "Data Loss Prevention (DLP)",
        "Database Activity Monitoring",
        "Data-at-Rest Encryption",
        "Data-in-Transit Encryption",
        "Key Management System",
        "Backup & Restore Controls",
    ],
    "Application Security & DevSecOps": [
        "Secure Software Development Lifecycle (SSDLC)",
        "Static Application Security Testing (SAST)",
        "Dynamic Application Security Testing (DAST)",
        "Web Application Security Testing",
        "API Security Management",
        "Secure Code Review",
        "DevSecOps Pipeline Integration",
    ],
    "Security Monitoring & Analytics": [
        "Security Information & Event Management (SIEM)",
        "Security Operations Center (SOC)",
        "User & Entity Behavior Analytics (UEBA)",
        "Threat Intelligence Integration",
        "Anomaly Detection Analytics",
        "Security Use Case & Rule Management",
    ],
    "Incident Response & Forensics": [
        "Incident Response Plan",
        "Incident Response Runbooks",
        "Security Incident Logging & Tracking",
        "Digital Forensics Capability",
        "Post-Incident Lessons Learned",
    ],
    "Business Continuity & DR": [
        "Disaster Recovery Plan (DRP)",
        "Business Continuity Plan (BCP)",
        "Regular DR Drills & Testing",
        "Backup Site / DR Site Readiness",
        "RPO/RTO Definition & Monitoring",
    ],
    "Cloud Security": [
        "Cloud Security Posture Management (CSPM)",
        "Cloud Workload Protection Platform (CWPP)",
        "Cloud Access Security Broker (CASB)",
        "Cloud Configuration Hardening",
        "Cloud Identity & Access Controls",
    ],
    "OT / ICS Security": [
        "OT Network Segmentation",
        "OT Firewall & Security Gateway",
        "OT Asset Discovery",
        "OT Anomaly Detection",
        "OT Incident Response Plan",
        "Secure Remote Access to OT Systems",
    ],
    "Compliance & Audit": [
        "Regulatory Compliance Mapping (NCA/SAMA/CMA)",
        "Internal Cybersecurity Audit Program",
        "External Compliance Assessments",
        "Policy & Control Review Cycle",
    ],
    "Fraud & Trading Surveillance": [
        "Anti-Fraud Monitoring Platform",
        "Trading Surveillance System",
        "Market Abuse Detection Analytics",
        "Customer Transaction Monitoring",
    ],
    "Awareness & Human Risk": [
        "Cybersecurity Training Program",
        "Phishing Simulation Campaigns",
        "Insider Threat Monitoring Program",
    ],
}

# Regulator -> control categories (which groups apply)
REGULATOR_CONTROLS = {
    "NCA": [
        "Governance & Strategy",
        "Risk Management",
        "Asset & Configuration Management",
        "Identity & Access Management",
        "Network Security",
        "Endpoint & Server Security",
        "Data Protection & Privacy",
        "Application Security & DevSecOps",
        "Security Monitoring & Analytics",
        "Incident Response & Forensics",
        "Business Continuity & DR",
        "Cloud Security",
        "OT / ICS Security",
        "Compliance & Audit",
        "Awareness & Human Risk",
    ],
    "SAMA": [
        "Governance & Strategy",
        "Risk Management",
        "Identity & Access Management",
        "Network Security",
        "Endpoint & Server Security",
        "Data Protection & Privacy",
        "Application Security & DevSecOps",
        "Security Monitoring & Analytics",
        "Incident Response & Forensics",
        "Business Continuity & DR",
        "Cloud Security",
        "Compliance & Audit",
        "Fraud & Trading Surveillance",
        "Awareness & Human Risk",
    ],
    "CMA": [
        "Governance & Strategy",
        "Risk Management",
        "Identity & Access Management",
        "Network Security",
        "Endpoint & Server Security",
        "Data Protection & Privacy",
        "Application Security & DevSecOps",
        "Security Monitoring & Analytics",
        "Incident Response & Forensics",
        "Compliance & Audit",
        "Fraud & Trading Surveillance",
    ],
    "General": [
        "Governance & Strategy",
        "Risk Management",
        "Identity & Access Management",
        "Network Security",
        "Endpoint & Server Security",
        "Data Protection & Privacy",
        "Security Monitoring & Analytics",
        "Incident Response & Forensics",
        "Business Continuity & DR",
        "Awareness & Human Risk",
    ],
}

# Expected "next wave" controls per regulator (prediction of upcoming requirements)
EXPECTED_NEW_CONTROLS = {
    "NCA": [
        "Zero-Trust Architecture Adoption",
        "Supply Chain Cybersecurity Program",
        "Advanced Threat Hunting Capability",
        "Cloud CSPM Expansion",
        "Cloud CWPP for Critical Workloads",
    ],
    "SAMA": [
        "Advanced Anti-Fraud Analytics",
        "Customer Identity Protection Platform",
        "Continuous Controls Monitoring (CCM)",
        "Cloud Security Posture Management for Financial Services",
    ],
    "CMA": [
        "AI-Based Market Abuse Detection",
        "Next-Generation Trading Surveillance Analytics",
        "Enhanced Data Governance & Lineage Tools",
    ],
    "General": [
        "Zero-Trust Strategy",
        "Cloud Security Modernization",
        "Enhanced Endpoint XDR Visibility",
    ],
}

# Average SAR spend per missing control (rough business rule)
AVERAGE_CONTROL_COST_SAR = 200000.0


def get_regulator_control_categories(regulator: str):
    return REGULATOR_CONTROLS.get(regulator, REGULATOR_CONTROLS["General"])


def get_expected_new_controls(regulator: str):
    return EXPECTED_NEW_CONTROLS.get(regulator, EXPECTED_NEW_CONTROLS["General"])


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

            # SAFE numeric conversion for Predicted_Final_Value
            pred_val_raw = pipeline["Predicted_Final_Value"].copy()

            pred_val = (
                pred_val_raw
                .apply(lambda x: x[0] if isinstance(x, list) else x)  # extract from list
            )

            pred_val = pd.to_numeric(pred_val, errors="coerce").fillna(
                pipeline["Expected Value (SAR)"]
            )

            # SAFE numeric conversion for Predicted_Win_Prob
            pred_prob_raw = pipeline["Predicted_Win_Prob"].copy()

            pred_prob = (
                pred_prob_raw
                .apply(lambda x: x[0] if isinstance(x, list) else x)  # extract from list
            )

            pred_prob = pd.to_numeric(pred_prob, errors="coerce").fillna(0.5)

            # Finally multiply
            pipeline_expected_value = (pred_val * pred_prob).sum()

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
        top_loss_stage = next(iter.summary["dominant_stage_lost"].keys())
        insights.append(
            f"Most losses happen around the '{top_loss_stage}' stage; investigate what typically goes wrong at this point."
        )

    summary["insights"] = insights
    return summary


# =========================================================
# SECTOR DETECTION (AUTO-DETECTION IF MISSING)
# =========================================================

def infer_sector(customer_name: str):
    """
    Auto-detect customer sector based on name patterns.
    """
    name = str(customer_name).lower()

    banking_kw = ["bank", "al rajhi", "riyad", "inma", "samba", "arab national"]
    fintech_kw = ["payment", "fintech", "pay", "wallet"]
    cma_kw = ["capital market", "brokerage", "tadawul", "investment", "trading"]
    gov_kw = ["ministry", "gov", "municipality", "university", "authority"]
    critical_kw = ["aramco", "electricity", "oil", "gas", "industrial", "petro"]
    health_kw = ["hospital", "medical", "clinic"]
    edu_kw = ["university", "college", "school"]

    if any(kw in name for kw in banking_kw):
        return "Banking"
    if any(kw in name for kw in fintech_kw):
        return "Fintech"
    if any(kw in name for kw in cma_kw):
        return "Capital Markets"
    if any(kw in name for kw in gov_kw):
        return "Government"
    if any(kw in name for kw in critical_kw):
        return "Critical Infrastructure"
    if any(kw in name for kw in health_kw):
        return "Healthcare"
    if any(kw in name for kw in edu_kw):
        return "Education"

    return "Private"


# =========================================================
# IMPLEMENTED CONTROLS (LEARNED FROM HISTORICAL WINS)
# =========================================================

def infer_implemented_controls(df, customer):
    """
    Infer implemented controls from previously WON opportunities for that customer.
    This is approximate but useful:
    - If customer previously bought SIEM → assume SIEM implemented
    - If bought EDR → assume EDR present
    - etc.
    """

    cust_data = df[(df["Customer Name"] == customer) &
                   (df["Status_Simplified"] == "Won")].copy()

    if cust_data.empty:
        return []

    solution_text = " ".join(
        str(x).lower() for x in cust_data["Solution Type"].fillna("")
    )

    mapping = {
        "siem": "Security Information & Event Management (SIEM)",
        "waf": "Web Application Firewall (WAF)",
        "firewall": "Next-Generation Firewall (NGFW)",
        "ngfw": "Next-Generation Firewall (NGFW)",
        "edr": "Endpoint Detection & Response (EDR)",
        "xdr": "Extended Detection & Response (XDR)",
        "dlt": "Data Loss Prevention (DLP)",
        "dlp": "Data Loss Prevention (DLP)",
        "idm": "Identity Lifecycle Management",
        "pam": "Privileged Access Management (PAM)",
        "mfa": "Multi-Factor Authentication (MFA)",
        "soc": "Security Operations Center (SOC)",
        "ti": "Threat Intelligence Integration",
        "backup": "Backup & Restore Controls",
        "dr": "Disaster Recovery Plan (DRP)",
    }

    found = []
    for kw, ctrl in mapping.items():
        if kw in solution_text:
            found.append(ctrl)

    return list(set(found))


# =========================================================
# REGULATORY BUSINESS SUMMARY (USED BEFORE FORECAST)
# =========================================================

def regulatory_business_summary(df):
    """
    Builds summary of historical revenue by regulator.
    """
    summary_list = []

    for cust, grp in df.groupby("Customer Name"):

        regs = grp.explode("Regulatory_Drivers")["Regulatory_Drivers"].value_counts().to_dict()
        revenue = grp[grp["Status_Simplified"] == "Won"]["Final Contract Value (SAR)"].sum()

        summary_list.append({
            "Customer Name": cust,
            "Regulatory_Driver_Counts": regs,
            "Historical_Revenue_SAR": revenue,
        })

    return pd.DataFrame(summary_list)


# =========================================================
# REGULATORY FORECAST ENGINE
# =========================================================

def build_regulatory_forecast(df):
    """
    For each customer:
    - Infer sector
    - Map regulator
    - Required categories
    - Full control list
    - Implemented controls
    - Missing controls
    - Expected new controls
    - Estimated new spend
    """

    result_rows = []

    for cust, grp in df.groupby("Customer Name"):

        sector = infer_sector(cust)
        regulator = SECTOR_REGULATOR_MAP.get(sector, "General")

        required_categories = get_regulator_control_categories(regulator)

        # Full list of required controls (all controls inside those categories)
        required_full = []
        for cat in required_categories:
            required_full.extend(CONTROL_LIBRARY.get(cat, []))

        required_full = list(sorted(set(required_full)))

        implemented = infer_implemented_controls(df, cust)

        missing = sorted(set(required_full) - set(implemented))

        expected_future = get_expected_new_controls(regulator)

        estimated_value = len(missing) * AVERAGE_CONTROL_COST_SAR

        result_rows.append({
            "Customer Name": cust,
            "Sector": sector,
            "Regulator": regulator,
            "Required_Control_Categories": required_categories,
            "Required_All_Controls": required_full,
            "Implemented_Controls": implemented,
            "Missing_Controls": missing,
            "Expected_New_Controls": expected_future,
            "Missing_Control_Count": len(missing),
            "Expected_Spend_SAR": estimated_value,
        })

    return pd.DataFrame(result_rows)


# =========================================================
# MAIN ANALYSIS FUNCTION
# =========================================================

def run_full_analysis(df):
    """
    Preprocessing, ML predictions, gap detection, KPI calculation, regulatory forecast.
    """
    df = df.copy()

    # Date parsing
    df["Expected Closing Date ( Deadline)"] = parse_date(
        df["Expected Closing Date ( Deadline)"]
    )
    df["Proposal Submission Date ( TP sent to BDO)"] = parse_date(
        df["Proposal Submission Date ( TP sent to BDO)"]
    )
    df["Last Updated Date"] = parse_date(df["Last Updated Date"])

    today = pd.Timestamp.today()

    df["days_to_deadline"] = (
        df["Expected Closing Date ( Deadline)"] - today
    ).dt.days

    df["days_since_last_update"] = (
        today - df["Last Updated Date"]
    ).dt.days

    df["Status_Simplified"] = df["Opportunity Status"].apply(map_status)

    # Regulatory tag
    df["Regulatory_Drivers"] = df.apply(tag_regulatory_drivers, axis=1)

    # GAP detection
    df = detect_business_gaps(df, today=today)

    # Simplify ML columns
    ml_df = df.copy()
    ml_df["target_win"] = ml_df["Status_Simplified"].apply(
        lambda x: 1 if x == "Won" else 0
    )
    ml_df["regression_target"] = ml_df["Final Contract Value (SAR)"]

    # ML Classification Model
    features = [
        "Solution Size Type",
        "Opportunity Stage",
        "Expected Value (SAR)",
        "days_to_deadline",
    ]

    X = ml_df[features]
    y = ml_df["target_win"]

    num_cols = ["Expected Value (SAR)", "days_to_deadline"]
    cat_cols = ["Solution Size Type", "Opportunity Stage"]

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)]
    )

    clf = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf.fit(X_train, y_train)
        pred_prob = clf.predict_proba(X)[:, 1]
        df["Predicted_Win_Prob"] = pred_prob
    except Exception:
        df["Predicted_Win_Prob"] = 0.5

    # ML Regression Model
    reg_y = ml_df["regression_target"].fillna(ml_df["Expected Value (SAR)"])
    reg_X = ml_df[features]

    reg = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
    ])

    try:
        reg.fit(reg_X, reg_y)
        pred_val = reg.predict(reg_X)
        df["Predicted_Final_Value"] = pred_val
    except Exception:
        df["Predicted_Final_Value"] = df["Expected Value (SAR)"]

    # Customer KPIs
    customer_kpis = build_customer_kpis(df)

    # Regulatory Forecast
    regulatory_forecast = build_regulatory_forecast(df)

    return df, customer_kpis, regulatory_forecast


# =========================================================
# STREAMLIT APPLICATION UI
# =========================================================

st.title("AI-Driven Smart Business Opportunity Analyzer")

uploaded = st.file_uploader("Upload Opportunity Excel/CSV Dataset", type=["xls", "xlsx", "csv"])

if uploaded is not None:

    df = pd.read_excel(uploaded) if uploaded.name.endswith(("xls", "xlsx")) else pd.read_csv(uploaded)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df_processed, customer_kpis, regulatory_forecast = run_full_analysis(df)

    tabs = st.tabs([
        "📊 Data Overview",
        "🤖 Opportunities AI Predictions",
        "🧭 Customer Insights",
        "⚖ Regulatory View",
        "📈 Regulatory Revenue Forecast",
    ])

    # =====================================================================
    # TAB 1 — DATA OVERVIEW
    # =====================================================================

    with tabs[0]:
        st.subheader("Dataset Summary")
        st.write(df_processed.head(20))

        st.subheader("Value Distribution")
        fig_val = px.histogram(df_processed, x="Expected Value (SAR)", title="Expected Value Distribution")
        st.plotly_chart(fig_val, use_container_width=True)

        fig_fin = px.histogram(df_processed, x="Final Contract Value (SAR)", title="Final Contract Values")
        st.plotly_chart(fig_fin, use_container_width=True)

    # =====================================================================
    # TAB 2 — OPPORTUNITIES AI
    # =====================================================================

    with tabs[1]:
        st.subheader("Predicted Win Probability Distribution")
        fig_prob = px.histogram(df_processed, x="Predicted_Win_Prob")
        st.plotly_chart(fig_prob, use_container_width=True)

        st.subheader("Expected vs Predicted Final Value")
        fig_scatter = px.scatter(
            df_processed, x="Expected Value (SAR)",
            y="Predicted_Final_Value",
            color="Predicted_Win_Prob",
            hover_data=["Customer Name"],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # =====================================================================
    # TAB 3 — CUSTOMER INSIGHTS
    # =====================================================================

    with tabs[2]:
        st.subheader("Customer KPIs Overview")
        st.dataframe(customer_kpis)

        selected_customer = st.selectbox("Select Customer", customer_kpis["Customer Name"].unique())

        if selected_customer:
            st.subheader(f"Win/Loss Summary for {selected_customer}")
            insights = explain_customer_win_loss(selected_customer, df_processed)
            st.write(insights)

    # =====================================================================
    # TAB 4 — REGULATORY VIEW
    # =====================================================================

    with tabs[3]:
        st.subheader("Regulatory Drivers per Opportunity")
        reg_count_df = (
            df_processed.explode("Regulatory_Drivers")["Regulatory_Drivers"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Regulator", "Regulatory_Drivers": "Count"})
        )

        fig_regs = px.bar(reg_count_df, x="Regulator", y="Count", title="Regulatory Keywords Identified")
        st.plotly_chart(fig_regs, use_container_width=True)

    # =====================================================================
    # TAB 5 — REGULATORY REVENUE FORECAST (FULL HYBRID VIEW)
    # =====================================================================

    with tabs[4]:
        st.header("📈 Regulatory Revenue Forecast")

        st.subheader("Forecast Table")
        st.dataframe(regulatory_forecast)

        # ---- High-demand controls (market demand) ----
        st.subheader("🔥 Top Missing Controls Across All Customers")

        missing_exploded = regulatory_forecast.explode("Missing_Controls")
        missing_counts = (
            missing_exploded["Missing_Controls"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Control", "Missing_Controls": "Count"})
        )

        fig_demand = px.bar(
            missing_counts.head(20),
            x="Control",
            y="Count",
            title="Top 20 Security Controls in Global Demand",
            text="Count",
        )
        fig_demand.update_traces(textposition="outside")
        fig_demand.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_demand, use_container_width=True)

        # ---- Detailed Drill-Down per Customer ----
        st.subheader("🔍 Detailed Regulatory View per Customer")

        selected_customer = st.selectbox("Select Customer", regulatory_forecast["Customer Name"].unique())

        if selected_customer:
            row = regulatory_forecast[regulatory_forecast["Customer Name"] == selected_customer].iloc[0]

            st.markdown(f"### {selected_customer}")
            st.write(f"**Sector:** {row['Sector']}")
            st.write(f"**Regulator:** {row['Regulator']}")
            st.write(f"**Expected New Spend (SAR):** {fmt_number(row['Expected_Spend_SAR'])}")

            st.markdown("### Required Control Categories")
            st.write(row["Required_Control_Categories"])

            # Expander for full detailed lists
            with st.expander("Full Required Controls (80+ items)"):
                st.write(row["Required_All_Controls"])

            with st.expander("Implemented Controls"):
                st.write(row["Implemented_Controls"])

            with st.expander("Missing Controls (High Demand)"):
                st.write(row["Missing_Controls"])

            with st.expander("Expected New Controls (Upcoming Regulatory Requirements)"):
                st.write(row["Expected_New_Controls"])

st.success("App Ready 🎉 – Upload your dataset or test your regulatory intelligence.")
