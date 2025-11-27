# AI-Driven Smart Business Opportunity Analyzer – Enterprise Edition

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =========================================================
# UNIVERSAL SAFE NUMERIC NORMALIZER
# =========================================================

def normalize_to_float(x, default: float = 0.0) -> float:
    """
    Convert ANY strange value into a usable float.
    Handles:
      - lists/tuples → mean (or first if single element)
      - strings → numeric if possible
      - None/NaN → default
      - errors → default
    """
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            x = list(x)
            if len(x) == 0:
                return default
            if len(x) == 1:
                return float(x[0])
            return float(sum(x) / len(x))

        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default

        return float(x)
    except Exception:
        return default


# =========================================================
# STREAMLIT PAGE CONFIG & THEME
# =========================================================

st.set_page_config(
    page_title="AI-Driven Smart Business Opportunity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light corporate theme for Plotly
px.defaults.template = "plotly_white"

# CSS styling
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
    .metric-card {
        background: #ffffff;
        padding: 0.8rem 1rem;
        border-radius: 0.7rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar debug toggle
debug_mode = st.sidebar.checkbox("🔍 Debug mode", value=False)


# =========================================================
# CONSTANTS & KNOWLEDGE BASE
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

# Sector → primary regulator
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

# --- CONTROL LIBRARY (grouped by category) ---

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

AVERAGE_CONTROL_COST_SAR = 200_000.0


def get_regulator_control_categories(reg: str):
    return REGULATOR_CONTROLS.get(reg, REGULATOR_CONTROLS["General"])


def get_expected_new_controls(reg: str):
    return EXPECTED_NEW_CONTROLS.get(reg, EXPECTED_NEW_CONTROLS["General"])


# =========================================================
# BASIC HELPERS
# =========================================================

def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def map_status(status):
    if pd.isna(status):
        return np.nan
    s = str(status).strip().lower()
    if "won" in s or "success" in s:
        return "Won"
    if "lost" in s or "closed lost" in s or "cancel" in s:
        return "Lost"
    return "In Progress"


def tag_regulatory_drivers(row: pd.Series):
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
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return x


def format_currency_columns(df: pd.DataFrame, cols):
    df_disp = df.copy()
    for c in cols:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].apply(
                lambda v: fmt_number(v) if pd.notna(v) else ""
            )
    return df_disp


# =========================================================
# GAP DETECTION
# =========================================================

def categorize_gap(reason: str) -> str:
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


def detect_business_gaps(df: pd.DataFrame, today=None) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.today()

    data = df.copy()
    reasons = []

    for _, row in data.iterrows():
        row_reasons = []

        ev = row.get("Expected Value (SAR)", np.nan)
        try:
            high_value = pd.notna(ev) and float(ev) > 100_000
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

        proposal_date = row.get("Proposal Submission Date ( TP sent to BDO)")
        if pd.isna(proposal_date) and pd.notna(days_to_deadline) and days_to_deadline <= 0:
            row_reasons.append(
                "Deadline reached/passed but no proposal submission date recorded."
            )

        days_since_update = row.get("days_since_last_update", np.nan)
        if (
            pd.notna(days_since_update)
            and days_since_update > 30
            and status == "In Progress"
        ):
            row_reasons.append(
                "Opportunity not updated for more than 30 days but still in progress."
            )

        expected_val = row.get("Expected Value (SAR)", np.nan)
        final_val = row.get("Final Contract Value (SAR)", np.nan)
        if pd.notna(expected_val) and pd.notna(final_val) and status in ["Won", "Lost"]:
            try:
                diff_ratio = abs(float(final_val) - float(expected_val)) / max(
                    float(expected_val), 1.0
                )
                if diff_ratio > 0.3:
                    row_reasons.append(
                        "Significant deviation between expected and final contract value."
                    )
            except Exception:
                pass

        reasons.append("; ".join(row_reasons) if row_reasons else "")

    data["Gap_Reason"] = reasons
    data["Gap_Flag"] = data["Gap_Reason"].apply(lambda x: 1 if x else 0)
    data["Gap_Category"] = data["Gap_Reason"].apply(categorize_gap)
    return data


# =========================================================
# CUSTOMER KPIs
# =========================================================

def build_customer_kpis(df: pd.DataFrame) -> pd.DataFrame:
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
            + np.log1p(normalize_to_float(monetary) / 100_000.0 + 1e-9)
        )

        pipeline = grp[inprog_mask].copy()
        if not pipeline.empty:
            pred_val = pipeline["Predicted_Final_Value"].apply(
                lambda v: normalize_to_float(v, default=0.0)
            )
            pred_prob = pipeline["Predicted_Win_Prob"].apply(
                lambda v: normalize_to_float(v, default=0.5)
            )
            pipeline_expected_value = float((pred_val * pred_prob).sum())
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


# =========================================================
# CUSTOMER WIN/LOSS EXPLANATION
# =========================================================

def explain_customer_win_loss(customer_name: str, df: pd.DataFrame) -> dict:
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
                f"Many losses are linked to {top_reg_loss}-driven projects; review alignment with that regulator’s controls."
            )

    loss_dict = summary.get("dominant_stage_lost", {})
    if isinstance(loss_dict, dict) and len(loss_dict) > 0:
        top_loss_stage = next(iter(loss_dict.keys()))
    else:
        top_loss_stage = "No dominant loss stage"

    summary["top_loss_stage"] = top_loss_stage
    insights.append(
        f"Most losses happen around the '{top_loss_stage}' stage; investigate typical blockers at this point."
    )

    summary["insights"] = insights
    return summary


# =========================================================
# SECTOR DETECTION & IMPLEMENTED CONTROLS
# =========================================================

def infer_sector(customer_name: str) -> str:
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


def infer_implemented_controls(df: pd.DataFrame, customer: str):
    cust_data = df[
        (df["Customer Name"] == customer)
        & (df["Status_Simplified"] == "Won")
    ].copy()

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
# REGULATORY FORECAST ENGINE
# =========================================================

def build_regulatory_forecast(df: pd.DataFrame) -> pd.DataFrame:
    result_rows = []

    for cust, grp in df.groupby("Customer Name"):
        sector = infer_sector(cust)
        regulator = SECTOR_REGULATOR_MAP.get(sector, "General")

        required_categories = get_regulator_control_categories(regulator)

        required_full = []
        for cat in required_categories:
            required_full.extend(CONTROL_LIBRARY.get(cat, []))
        required_full = sorted(set(required_full))

        implemented = infer_implemented_controls(df, cust)
        missing = sorted(set(required_full) - set(implemented))
        expected_future = get_expected_new_controls(regulator)
        estimated_value = len(missing) * AVERAGE_CONTROL_COST_SAR

        result_rows.append(
            {
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
            }
        )

    return pd.DataFrame(result_rows)


# =========================================================
# MAIN ANALYSIS PIPELINE
# =========================================================

def run_full_analysis(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Parse dates
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
    df["days_since_last_update"] = (today - df["Last Updated Date"]).dt.days

    df["Status_Simplified"] = df["Opportunity Status"].apply(map_status)
    df["Regulatory_Drivers"] = df.apply(tag_regulatory_drivers, axis=1)

    # Gap detection
    df = detect_business_gaps(df, today=today)

    # ML Preparation
    ml_df = df.copy()
    ml_df["target_win"] = ml_df["Status_Simplified"].apply(
        lambda x: 1 if x == "Won" else 0
    )
    ml_df["regression_target"] = ml_df["Final Contract Value (SAR)"]

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

    # Classification model
    clf = Pipeline(
        [
            ("pre", pre),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf.fit(X_train, y_train)
        pred_prob = clf.predict_proba(X)[:, 1]
        df["Predicted_Win_Prob"] = pred_prob
    except Exception as e:
        if debug_mode:
            st.warning(f"Classification model failed, using default 0.5. Error: {e}")
        df["Predicted_Win_Prob"] = 0.5

    # Regression model
    reg_y = ml_df["regression_target"].fillna(ml_df["Expected Value (SAR)"])
    reg_X = ml_df[features]

    reg = Pipeline(
        [
            ("pre", pre),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    try:
        reg.fit(reg_X, reg_y)
        pred_val = reg.predict(reg_X)
        df["Predicted_Final_Value"] = pred_val
    except Exception as e:
        if debug_mode:
            st.warning(
                f"Regression model failed, using Expected Value instead. Error: {e}"
            )
        df["Predicted_Final_Value"] = df["Expected Value (SAR)"]

    customer_kpis = build_customer_kpis(df)
    regulatory_forecast = build_regulatory_forecast(df)

    return df, customer_kpis, regulatory_forecast


# =========================================================
# STREAMLIT APP UI
# =========================================================

st.title("AI-Driven Smart Business Opportunity Analyzer")

st.markdown(
    "Upload your anonymised opportunity tracking sheet (Excel/CSV) to generate AI-driven insights."
)

uploaded = st.file_uploader(
    "Upload Opportunity Excel/CSV Dataset", type=["xls", "xlsx", "csv"]
)

if uploaded is None:
    st.info("👆 Please upload your dataset to start the analysis.")
    st.stop()

# Read file
if uploaded.name.endswith((".xls", ".xlsx")):
    df_raw = pd.read_excel(uploaded)
else:
    df_raw = pd.read_csv(uploaded)

missing_cols = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
if missing_cols:
    st.error(f"The following required columns are missing: {missing_cols}")
    st.stop()

df_processed, customer_kpis, regulatory_forecast = run_full_analysis(df_raw)

# Tabs layout
tabs = st.tabs(
    [
        "📊 Data Overview",
        "🤖 Opportunities AI Predictions",
        "🧭 Customer Insights",
        "⚖ Regulatory View",
        "📈 Regulatory Revenue Forecast",
    ]
)

# ---------------------------------------------------------
# TAB 1 – DATA OVERVIEW
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset Snapshot")

    money_cols = ["Expected Value (SAR)", "Final Contract Value (SAR)"]
    st.dataframe(format_currency_columns(df_processed.head(50), money_cols))

    # Quick metrics
    total_opps = len(df_processed)
    won = (df_processed["Status_Simplified"] == "Won").sum()
    lost = (df_processed["Status_Simplified"] == "Lost").sum()
    inprog = (df_processed["Status_Simplified"] == "In Progress").sum()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Opportunities", total_opps)
    with col_m2:
        st.metric("Won", won)
    with col_m3:
        st.metric("Lost", lost)
    with col_m4:
        st.metric("In Progress", inprog)

    st.markdown("### Value Distributions")
    c1, c2 = st.columns(2)

    with c1:
        try:
            fig_val = px.histogram(
                df_processed,
                x="Expected Value (SAR)",
                nbins=30,
                title="Expected Value Distribution",
            )
            st.plotly_chart(fig_val, use_container_width=True)
        except Exception as e:
            st.error(f"Could not draw Expected Value histogram: {e}")

    with c2:
        try:
            fig_fin = px.histogram(
                df_processed,
                x="Final Contract Value (SAR)",
                nbins=30,
                title="Final Contract Value Distribution",
            )
            st.plotly_chart(fig_fin, use_container_width=True)
        except Exception as e:
            st.error(f"Could not draw Final Contract histogram: {e}")

    # Export button
    st.download_button(
        "⬇ Download processed dataset (CSV)",
        data=df_processed.to_csv(index=False).encode("utf-8-sig"),
        file_name="processed_opportunities.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# TAB 2 – OPPORTUNITIES AI PREDICTIONS
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Predicted Win Probabilities")

    try:
        fig_prob = px.histogram(
            df_processed,
            x="Predicted_Win_Prob",
            nbins=20,
            title="Distribution of Predicted Win Probability",
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    except Exception as e:
        st.error(f"Could not draw probability histogram: {e}")

    st.subheader("Expected vs Predicted Final Value")
    try:
        fig_scatter = px.scatter(
            df_processed,
            x="Expected Value (SAR)",
            y="Predicted_Final_Value",
            color="Predicted_Win_Prob",
            hover_data=["Customer Name", "Opportunity Name"],
            title="Expected vs Predicted Final Value",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    except Exception as e:
        st.error(f"Could not draw Expected vs Predicted scatter: {e}")

# ---------------------------------------------------------
# TAB 3 – CUSTOMER INSIGHTS
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Customer Portfolio KPIs")

    if not customer_kpis.empty:
        display_kpis = customer_kpis.copy()
        display_kpis["Historical_Revenue_SAR"] = display_kpis[
            "Historical_Revenue_SAR"
        ].apply(fmt_number)
        display_kpis["Pipeline_Expected_Revenue_SAR"] = display_kpis[
            "Pipeline_Expected_Revenue_SAR"
        ].apply(fmt_number)
        st.dataframe(display_kpis)

        st.download_button(
            "⬇ Download customer KPIs (CSV)",
            data=customer_kpis.to_csv(index=False).encode("utf-8-sig"),
            file_name="customer_kpis.csv",
            mime="text/csv",
        )

        selected_customer = st.selectbox(
            "Select customer for detailed insights",
            customer_kpis["Customer Name"].unique(),
        )

        if selected_customer:
            summary = explain_customer_win_loss(selected_customer, df_processed)
            if summary:
                st.markdown(f"### Win/Loss Overview – {selected_customer}")

                cm1, cm2, cm3, cm4 = st.columns(4)
                with cm1:
                    st.metric("Total Opps", summary["total_opportunities"])
                with cm2:
                    st.metric("Won", summary["won_count"])
                with cm3:
                    st.metric("Lost", summary["lost_count"])
                with cm4:
                    st.metric(
                        "Win Rate",
                        f"{summary['win_rate']*100:.1f}%" if summary["win_rate"] == summary["win_rate"] else "N/A",
                    )

                st.markdown("#### Key Insights")
                for i in summary["insights"]:
                    st.write("• " + i)

                # Small charts for domains won/lost
                cw1, cw2 = st.columns(2)
                with cw1:
                    if summary["dominant_domains_won"]:
                        won_df = (
                            pd.Series(summary["dominant_domains_won"])
                            .reset_index()
                            .rename(columns={"index": "Domain", 0: "Count"})
                        )
                        fig_won_dom = px.bar(
                            won_df,
                            x="Domain",
                            y="Count",
                            title="Top Domains – Won",
                            text="Count",
                        )
                        fig_won_dom.update_traces(textposition="outside")
                        st.plotly_chart(fig_won_dom, use_container_width=True)

                with cw2:
                    if summary["dominant_domains_lost"]:
                        lost_df = (
                            pd.Series(summary["dominant_domains_lost"])
                            .reset_index()
                            .rename(columns={"index": "Domain", 0: "Count"})
                        )
                        fig_lost_dom = px.bar(
                            lost_df,
                            x="Domain",
                            y="Count",
                            title="Top Domains – Lost",
                            text="Count",
                        )
                        fig_lost_dom.update_traces(textposition="outside")
                        st.plotly_chart(fig_lost_dom, use_container_width=True)
    else:
        st.info("No customer-level KPIs available.")

# Ensure regulatory driver column exists and is list-like
    if "Regulatory_Drivers" not in df_processed.columns:
        df_processed["Regulatory_Drivers"] = [[] for _ in range(len(df_processed))]

# Normalize (convert strings, NaN to lists)
    df_processed["Regulatory_Drivers"] = df_processed["Regulatory_Drivers"].apply(
        lambda x: x if isinstance(x, list) else []
)

# ---------------------------------------------------------
# TAB 4 – REGULATORY VIEW
# ---------------------------------------------------------
with tabs[3]:

    st.header("⚖ Regulatory View")
    st.subheader("🧠 Smart Regulatory Insights")

    # ============================================================
    # 1️⃣ REGULATOR SUMMARY (FIXED & SAFE)
    # ============================================================
    st.subheader("🔍 Regulator Detection Summary")

    reg_summary = (
        df_processed["Regulatory_Drivers"]
        .explode()
        .dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Regulator", "Regulatory_Drivers": "Count"})
    )

    # Force Count numeric
    reg_summary["Count"] = pd.to_numeric(reg_summary["Count"], errors="coerce").fillna(0)

    total = reg_summary["Count"].sum() or 1
    reg_summary["Percentage"] = ((reg_summary["Count"] / total) * 100).round(1).astype(str) + "%"

    st.subheader("🔎 Regulator Detection Summary")
    st.dataframe(reg_summary)

    # ============================================================
    # 2️⃣ SMART INTERPRETATION
    # ============================================================
    st.subheader("🧠 Interpretation")

    unique_regs = reg_summary["Regulator"].tolist()

    if not unique_regs or unique_regs == ["General"]:
        st.info("""
        **All detected regulators = 'General'**

        This means:
        - No NCA cybersecurity terms detected  
        - No SAMA banking terminology detected  
        - No CMA investment/trading terminology found  

        To increase regulatory intelligence:
        - Add clearer descriptions in Opportunity Name  
        - Mention frameworks (NCA ECC, SAMA CSF, CMA)  
        - Use domain-specific keywords in Solution Type  
        """)
    else:
        st.success(f"Detected regulators: {', '.join(unique_regs)}")

    # ============================================================
    # 3️⃣ KEYWORD COVERAGE TABLE
    # ============================================================
    st.subheader("📚 Regulatory Keyword Coverage")

    keyword_coverage = reg_summary[["Regulator", "Count"]].rename(
        columns={"Regulator": "Keyword", "Count": "Frequency"}
    )
    st.dataframe(keyword_coverage, use_container_width=True)

    # ============================================================
    # 4️⃣ PIE CHART — Regulator Share Distribution
    # ============================================================
    st.subheader("📊 Regulator Share Distribution")

    try:
        if reg_summary.empty:
            st.info("No regulatory data available.")
        else:
            fig_pie = px.pie(
                reg_summary,
                names="Regulator",
                values="Count",
                title="Distribution of Regulatory Drivers",
                hole=0.35
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
    except Exception as e:
        st.error(f"Could not draw regulator share pie chart: {e}")

    # ============================================================
    # 5️⃣ BAR CHART — Regulator Frequency
    # ============================================================
    st.subheader("📈 Regulatory Driver Frequency")

    try:
        if reg_summary.empty:
            st.info("No regulatory drivers available.")
        else:
            fig_bar = px.bar(
                reg_summary,
                x="Regulator",
                y="Count",
                text="Count",
                title="Frequency of Regulatory Drivers",
                color="Regulator",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Could not draw regulatory bar chart: {e}")

    # ============================================================
    # 6️⃣ MOST FREQUENT REGULATORY KEYWORDS
    # ============================================================
    st.subheader("🔍 Most Frequent Regulatory Keywords")

    try:
        keyword_hits = []

        for _, row in df_processed.iterrows():
            text = (
                str(row.get("Opportunity Name", "")).lower() + " " +
                str(row.get("Domain", "")).lower() + " " +
                str(row.get("Solution Type", "")).lower()
            )

            for reg, kw_list in REGULATORY_KEYWORDS.items():
                for kw in kw_list:
                    if kw in text:
                        keyword_hits.append(kw)

        if not keyword_hits:
            st.info("No regulatory keywords detected in your dataset.")
        else:
            kw_df = (
                pd.Series(keyword_hits)
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Keyword", 0: "Frequency"})
            )

            fig_kw = px.bar(
                kw_df.head(15),
                x="Keyword",
                y="Frequency",
                text="Frequency",
                title="Most Common Regulatory Keywords",
                color="Frequency"
            )
            fig_kw.update_traces(textposition="outside")
            fig_kw.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_kw, use_container_width=True)

    except Exception as e:
        st.error(f"Could not draw keyword analysis: {e}")



# ---------------------------------------------------------
# TAB 5 – REGULATORY REVENUE FORECAST
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("🔍 Detailed Regulatory View per Customer")
    
    selected_customer = st.selectbox(
        "Select Customer",
        regulatory_forecast["Customer Name"].unique(),
        key="reg_customer"
    )
    
    if selected_customer:
    
        row = regulatory_forecast[
            regulatory_forecast["Customer Name"] == selected_customer
        ].iloc[0]
    
        # ====== TOP KPI CARDS ======
        kpi1, kpi2, kpi3 = st.columns(3)
    
        kpi1.metric("Sector", row["Sector"])
        kpi2.metric("Regulator", row["Regulator"])
        kpi3.metric(
            "Expected New Spend (SAR)",
            fmt_number(row["Expected_Spend_SAR"])
        )
    
        st.markdown("---")
    
        # ====== DONUT CHART: Implemented vs Missing ======
        implemented_count = len(row["Implemented_Controls"])
        missing_count = len(row["Missing_Controls"])
        total_required = implemented_count + missing_count
    
        donut_df = pd.DataFrame({
            "Status": ["Implemented", "Missing"],
            "Count": [implemented_count, missing_count]
        })
    
        fig_donut = px.pie(
            donut_df,
            values="Count",
            names="Status",
            hole=0.55,
            color="Status",
            color_discrete_map={
                "Implemented": "#2ca02c",
                "Missing": "#d62728"
            },
            title="Overall Compliance Readiness"
        )
        fig_donut.update_layout(showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)
    
        st.markdown("---")
    
        # ====== CATEGORY-LEVEL BAR CHART ======
        categories = row["Required_Control_Categories"]
    
        implemented_map = {
            c: any(cn in c for cn in row["Implemented_Controls"])
            for c in categories
        }
        missing_map = {
            c: any(cn in c for cn in row["Missing_Controls"])
            for c in categories
        }
    
        bar_df = pd.DataFrame({
            "Category": categories,
            "Implemented": [1 if implemented_map[c] else 0 for c in categories],
            "Missing": [1 if missing_map[c] else 0 for c in categories],
        })
    
        fig_bar = px.bar(
            bar_df,
            x="Category",
            y=["Implemented", "Missing"],
            barmode="group",
            title="Control Readiness by Category",
        )
        fig_bar.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_bar, use_container_width=True)
    
        st.markdown("---")
    
        # ====== BADGE-STYLE LISTS ======
    
        st.markdown("### 🟦 Required Control Categories")
        for item in row["Required_Control_Categories"]:
            st.markdown(f"""
                <span style='background:#e8f0fe;padding:6px 12px;border-radius:6px;margin:3px;display:inline-block;color:#174ea6;'>
                {item}
                </span>
            """, unsafe_allow_html=True)
    
        st.markdown("### 🟩 Implemented Controls")
        if len(row["Implemented_Controls"]) == 0:
            st.info("No implemented controls detected.")
        else:
            for item in row["Implemented_Controls"]:
                st.markdown(f"""
                    <span style='background:#e8f5e9;padding:6px 12px;border-radius:6px;margin:3px;display:inline-block;color:#2e7d32;'>
                    {item}
                    </span>
                """, unsafe_allow_html=True)
    
        st.markdown("### 🟥 Missing Controls (High-Demand)")
        if len(row["Missing_Controls"]) == 0:
            st.success("No missing controls. Customer is fully compliant.")
        else:
            for item in row["Missing_Controls"]:
                st.markdown(f"""
                    <span style='background:#ffebee;padding:6px 12px;border-radius:6px;margin:3px;display:inline-block;color:#c62828;'>
                    {item}
                    </span>
                """, unsafe_allow_html=True)
    
        st.markdown("### 🟨 Expected New Controls (Upcoming Regulations)")
        for item in row["Expected_New_Controls"]:
            st.markdown(f"""
                <span style='background:#fff8e1;padding:6px 12px;border-radius:6px;margin:3px;display:inline-block;color:#ff8f00;'>
                {item}
                </span>
            """, unsafe_allow_html=True)
