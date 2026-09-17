import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)

# â”€â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="CustomerChurnPrediction",
    page_icon="ðŸ”®",
    layout="wide",
    initial_sidebar_state="expanded"
)

# â”€â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* â”€â”€ Color palette: navy (#0f172a) Â· indigo (#6366f1) Â· white (#ffffff) â”€â”€ */

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background-color: #0f172a; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0b1120 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 18px 22px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(99,102,241,0.2);
  }
  [data-testid="metric-container"] label {
    color: #94a3b8 !important; font-size: 0.8rem !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important;
  }

  /* Headers */
  h1 { color: #ffffff !important; font-size: 2.4rem !important; font-weight: 700 !important; }
  h2 { color: #e2e8f0 !important; font-weight: 600 !important; }
  h3 { color: #cbd5e1 !important; font-weight: 500 !important; }
  p, li, span, label { color: #94a3b8 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px; padding: 4px; gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important; border-radius: 8px;
    transition: all 0.2s; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: #ffffff !important;
  }

  /* Buttons */
  .stButton > button {
    background: #6366f1 !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 10px 24px !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    background: #4f46e5 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
  }

  /* Dataframes */
  .dataframe { background: rgba(255,255,255,0.03) !important; border-radius: 8px; }

  /* Info boxes */
  .stAlert { border-radius: 10px; border: none !important; }

  /* Section dividers */
  hr { border-color: rgba(255,255,255,0.07) !important; }

  /* Hero banner */
  .hero-banner {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px; padding: 2rem; margin-bottom: 2rem;
    text-align: center;
  }

  /* Prediction result */
  .pred-churn {
    background: rgba(99,102,241,0.08);
    border: 2px solid rgba(239,68,68,0.5); border-radius: 12px;
    padding: 1.5rem; text-align: center;
  }
  .pred-no-churn {
    background: rgba(99,102,241,0.08);
    border: 2px solid rgba(99,102,241,0.5); border-radius: 12px;
    padding: 1.5rem; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_PATH = os.path.join(os.path.dirname(__file__), "churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "customer_churn_pipeline.pkl")

# â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data(show_spinner="Loading datasetâ€¦")
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Clean placeholders
    df["joined_through_referral"] = df["joined_through_referral"].replace("?", np.nan)
    df["medium_of_operation"]     = df["medium_of_operation"].replace("?", np.nan)
    df["avg_frequency_login_days"] = df["avg_frequency_login_days"].replace("Error", np.nan)

    # Fix types
    df["joining_date"]             = pd.to_datetime(df["joining_date"], errors="coerce")
    df["avg_frequency_login_days"] = pd.to_numeric(df["avg_frequency_login_days"], errors="coerce")

    # Drop pure IDs
    df.drop(columns=["Unnamed: 0", "security_no"], inplace=True, errors="ignore")

    # Feature engineering
    df["has_referral_id"] = np.where(
        df["referral_id"].isna() | (df["referral_id"] == "xxxxxxxx"), 0, 1
    )
    df.drop(columns=["referral_id"], inplace=True, errors="ignore")

    df["joining_year"]      = df["joining_date"].dt.year
    df["joining_month"]     = df["joining_date"].dt.month
    df["joining_day"]       = df["joining_date"].dt.day
    df["joining_dayofweek"] = df["joining_date"].dt.dayofweek
    ref_date = df["joining_date"].max()
    df["customer_tenure_days"] = (ref_date - df["joining_date"]).dt.days

    last_visit_dt          = pd.to_datetime(df["last_visit_time"], format="%H:%M:%S", errors="coerce")
    df["last_visit_hour"]  = last_visit_dt.dt.hour

    def map_period(h):
        if pd.isna(h): return np.nan
        h = int(h)
        if 5  <= h < 12: return "morning"
        if 12 <= h < 17: return "afternoon"
        if 17 <= h < 21: return "evening"
        return "night"

    df["visit_period"] = df["last_visit_hour"].apply(map_period)
    df.drop(columns=["joining_date", "last_visit_time"], inplace=True, errors="ignore")

    # Suspicious negatives
    df.loc[df["days_since_last_login"] < 0, "days_since_last_login"] = np.nan

    return df


@st.cache_resource(show_spinner="Training modelsâ€¦ (first run only, grab a â˜•)")
def get_or_train_model(df):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    target = "churn_risk_score"
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scl", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_pipe, num_cols),
                               ("cat", cat_pipe, cat_cols)])

    model = Pipeline([
        ("preprocessor", prep),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=20,
                                          min_samples_split=2, min_samples_leaf=1,
                                          class_weight=None, random_state=42, n_jobs=-1))
    ])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_data(show_spinner="Evaluating modelsâ€¦")
def evaluate_all_models(_df):
    target = "churn_risk_score"
    X = _df.drop(columns=[target])
    y = _df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scl", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_pipe, num_cols),
                               ("cat", cat_pipe, cat_cols)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra Trees":          ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
        "AdaBoost":             AdaBoostClassifier(random_state=42),
    }

    results = []
    roc_data = {}
    for name, clf in classifiers.items():
        pipe = Pipeline([("preprocessor", prep), ("model", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_te, y_prob))
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred), 4),
            "Recall":    round(recall_score(y_te, y_pred), 4),
            "F1":        round(f1_score(y_te, y_pred), 4),
            "ROC-AUC":   round(roc_auc_score(y_te, y_prob), 4),
        })
    return pd.DataFrame(results).sort_values("F1", ascending=False), roc_data, X_te, y_te


# â”€â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("## ðŸ“Š CustomerChurnPrediction")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["ðŸ  Overview", "ðŸ“Š Data Explorer", "ðŸ“ˆ EDA & Insights",
         "ðŸ¤– Model Comparison", "ðŸŽ¯ Live Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    **Dataset:** `churn.csv`  
    **Rows:** 36,992  
    **Features:** 24  
    **Target:** `churn_risk_score`
    """)
    st.markdown("---")
    st.markdown("Built by **Samhita Reddy** ðŸš€")

# â”€â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = load_and_preprocess()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: OVERVIEW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "ðŸ  Overview":
    st.markdown("""
    <div class="hero-banner">
      <h1 style="margin-bottom:0.3rem">ðŸ“Š CustomerChurnPrediction</h1>
      <p style="font-size:1.1rem; color:#c4b5fd !important">
        End-to-end Customer Churn Prediction Platform
      </p>
      <p style="color:#9ca3af !important; font-size:0.9rem">
        Powered by Random Forest Â· Built from CustomerChurn Notebook
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    churn_pct = round(df["churn_risk_score"].mean() * 100, 1)
    with col1:
        st.metric("ðŸ“‹ Total Records",  f"{len(df):,}")
    with col2:
        st.metric("ðŸ“‰ Churned (%)",     f"{churn_pct}%")
    with col3:
        st.metric("ðŸ”¢ Features",        "24")
    with col4:
        st.metric("ðŸŽ¯ Target",          "churn_risk_score")

    st.markdown("---")
    st.subheader("ðŸ“Œ Project Pipeline")
    cols = st.columns(5)
    steps = [
        ("1ï¸âƒ£ Data Ingestion",   "Load churn.csv (36,992 rows)"),
        ("2ï¸âƒ£ Cleaning",         "Handle nulls, dirty placeholders, type errors"),
        ("3ï¸âƒ£ Feature Engg.",    "Date decomposition, referral, visit period"),
        ("4ï¸âƒ£ Modeling",         "6 classifiers compared via F1 & ROC-AUC"),
        ("5ï¸âƒ£ Deployment",       "Live Streamlit predictor"),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
                        border-radius:14px; padding:1rem; text-align:center; height:120px;">
              <b style="color:#a78bfa">{title}</b><br>
              <small style="color:#9ca3af">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("ðŸ—ºï¸ Dataset Schema")
    schema = {
        "Column": ["age","gender","region_category","membership_category","joining_date",
                   "joined_through_referral","preferred_offer_types","medium_of_operation",
                   "internet_option","days_since_last_login","avg_time_spent",
                   "avg_transaction_value","avg_frequency_login_days","points_in_wallet",
                   "used_special_discount","offer_application_preference","past_complaint",
                   "complaint_status","feedback","churn_risk_score"],
        "Type":   ["int","cat","cat","cat","date","cat","cat","cat","cat",
                   "int","float","float","float","float","cat","cat","cat","cat","cat","int (target)"],
        "Notes":  ["18-65","M/F","City/Town/Village","Platinum/Premium/â€¦","Join date",
                   "Yes/No/?","Gift/Credit/No","Desktop/Mobile/Both","Fiber/WiFi/â€¦","Days",
                   "Minutes","â‚¹","Days","Points","Yes/No","Yes/No","Yes/No",
                   "Solved/Unsolved/â€¦","Positive/Negative/â€¦","0=retain, 1=churn"],
    }
    st.dataframe(pd.DataFrame(schema), use_container_width=True, height=500)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: DATA EXPLORER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ“Š Data Explorer":
    st.title("ðŸ“Š Data Explorer")

    tab1, tab2, tab3 = st.tabs(["ðŸ” Raw Sample", "ðŸ“ Statistics", "âŒ Missing Values"])

    with tab1:
        n = st.slider("Rows to show", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)

    with tab2:
        num_df = df.select_dtypes(include=["int64", "float64"])
        st.dataframe(num_df.describe().T.round(3), use_container_width=True)

    with tab3:
        miss = pd.DataFrame({
            "Missing": df.isna().sum(),
            "% Missing": (df.isna().sum() / len(df) * 100).round(2)
        }).query("Missing > 0").sort_values("Missing", ascending=False)

        if miss.empty:
            st.success("âœ… No missing values after preprocessing!")
        else:
            fig = px.bar(miss.reset_index(), x="index", y="Missing",
                         color="% Missing", color_continuous_scale="Blues",
                         labels={"index":"Column"},
                         title="Missing Values per Column")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss, use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: EDA & INSIGHTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ“ˆ EDA & Insights":
    st.title("ðŸ“ˆ EDA & Insights")

    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸŽ¯ Target Distribution",
        "ðŸ“Š Categorical Analysis",
        "ðŸ”¢ Numeric Analysis",
        "ðŸŒ¡ï¸ Correlation Heatmap"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            vc = df["churn_risk_score"].value_counts().reset_index()
            vc.columns = ["Label", "Count"]
            vc["Label"] = vc["Label"].map({0: "Retained", 1: "Churned"})
            fig = px.pie(vc, names="Label", values="Count",
                         color_discrete_sequence=["#6366f1","#1e293b"],
                         title="Churn Distribution", hole=0.45)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
                               legend_font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### ðŸ“Œ Key Stats")
            total    = len(df)
            churned  = int(df["churn_risk_score"].sum())
            retained = total - churned
            st.metric("Total Customers",  f"{total:,}")
            st.metric("Churned",           f"{churned:,} ({churned/total*100:.1f}%)")
            st.metric("Retained",          f"{retained:,} ({retained/total*100:.1f}%)")
            st.info("Slight class imbalance â€” churned customers slightly outnumber retained ones.")

    with tab2:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if "churn_risk_score" in cat_cols:
            cat_cols.remove("churn_risk_score")
        sel = st.selectbox("Choose categorical column", cat_cols)

        col1, col2 = st.columns(2)
        with col1:
            vc = df[sel].value_counts().head(15).reset_index()
            vc.columns = ["Category", "Count"]
            fig = px.bar(vc, x="Category", y="Count", color="Count",
                         color_continuous_scale="Blues",
                         title=f"Distribution: {sel}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e0e0e0", xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            churn_rate = df.groupby(sel)["churn_risk_score"].mean().reset_index()
            churn_rate.columns = ["Category", "ChurnRate"]
            churn_rate = churn_rate.sort_values("ChurnRate", ascending=False)
            fig = px.bar(churn_rate, x="Category", y="ChurnRate",
                         color="ChurnRate", color_continuous_scale="Blues",
                         title=f"Churn Rate by {sel}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e0e0e0", xaxis_tickangle=-35,
                               yaxis_tickformat=".0%")
            fig.update_traces(text=[f"{v:.1%}" for v in churn_rate["ChurnRate"]],
                              textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        if "churn_risk_score" in num_cols:
            num_cols.remove("churn_risk_score")
        sel = st.selectbox("Choose numeric column", num_cols)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=sel, color="churn_risk_score",
                               color_discrete_map={0:"#6366f1", 1:"#ffffff"},
                               barmode="overlay", marginal="violin",
                               title=f"Distribution: {sel} by Churn")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x="churn_risk_score", y=sel,
                         color="churn_risk_score",
                         color_discrete_map={0:"#6366f1", 1:"#ffffff"},
                         title=f"Boxplot: {sel} vs Churn",
                         labels={"churn_risk_score":"Churned"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include=["int64","float64"]).dropna()
        corr   = num_df.corr()
        fig = px.imshow(corr, color_continuous_scale="Blues", aspect="auto",
                        title="Feature Correlation Heatmap",
                        text_auto=".2f")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e0e0", height=600)
        st.plotly_chart(fig, use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: MODEL COMPARISON
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ¤– Model Comparison":
    st.title("ðŸ¤– Model Comparison")
    st.info("Training 6 classifiers on an 80/20 split. This may take a momentâ€¦")

    results_df, roc_data, X_te, y_te = evaluate_all_models(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸ“‹ Leaderboard", "ðŸ“Š Metric Charts", "ðŸ“‰ ROC Curves", "ðŸ”¥ Confusion Matrix"
    ])

    with tab1:
        st.dataframe(
            results_df.style.background_gradient(cmap="plasma", subset=["F1","ROC-AUC"])
                            .format("{:.4f}", subset=["Accuracy","Precision","Recall","F1","ROC-AUC"]),
            use_container_width=True
        )
        best = results_df.iloc[0]
        st.success(f"ðŸ† Best model: **{best['Model']}** â€” F1: {best['F1']:.4f} | ROC-AUC: {best['ROC-AUC']:.4f}")

    with tab2:
        melted = results_df.melt(id_vars="Model",
                                  value_vars=["Accuracy","Precision","Recall","F1","ROC-AUC"],
                                  var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#6366f1","#a5b4fc","#e2e8f0","#4f46e5","#c7d2fe","#818cf8"],
                     title="All Models â€” All Metrics")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e0e0", xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        colors = px.colors.qualitative.Vivid
        for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                                     line=dict(color=colors[i % len(colors)], width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                  line=dict(dash="dash", color="#6b7280")))
        fig.update_layout(
            title="ROC Curves â€” All Models",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", legend_font_color="#e0e0e0",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        sel_model = st.selectbox("Pick a model", results_df["Model"].tolist())
        # Re-train selected model for CM
        target = "churn_risk_score"
        X = df.drop(columns=[target])
        y = df[target]
        num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),("scl", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))])
        prep = ColumnTransformer([("num", num_pipe, num_cols),("cat", cat_pipe, cat_cols)])
        clf_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
            "AdaBoost":            AdaBoostClassifier(random_state=42),
        }
        X_tr, X_te2, y_tr, y_te2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe = Pipeline([("preprocessor", prep), ("model", clf_map[sel_model])])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te2)
        cm = confusion_matrix(y_te2, y_pred)

        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Not Churned","Churned"], y=["Not Churned","Churned"],
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix â€” {sel_model}")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.code(classification_report(y_te2, y_pred,
                target_names=["Not Churned","Churned"]))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: LIVE PREDICTOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸŽ¯ Live Predictor":
    st.title("ðŸŽ¯ Live Churn Predictor")
    st.markdown("Fill in a customer's profile and get an instant churn probability.")

    model = get_or_train_model(df)

    with st.form("prediction_form"):
        st.subheader("ðŸ‘¤ Customer Profile")

        col1, col2, col3 = st.columns(3)
        with col1:
            age    = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["M", "F"])
            region = st.selectbox("Region Category", ["City","Town","Village","Unknown"])
        with col2:
            membership = st.selectbox("Membership Category",
                ["No Membership","Basic Membership","Silver Membership",
                 "Gold Membership","Premium Membership","Platinum Membership"])
            offer_type = st.selectbox("Preferred Offer Type",
                ["Gift Vouchers/Coupons","Credit/Debit Card Offers","Without Offers","Unknown"])
            medium = st.selectbox("Medium of Operation", ["Desktop","Mobile","Both","?"])
        with col3:
            internet = st.selectbox("Internet Option", ["Wi-Fi","Mobile_Data","Fiber_Optic"])
            days_login = st.slider("Days Since Last Login", 0, 30, 5)
            avg_time   = st.slider("Avg Time Spent (min)", 0.0, 1000.0, 300.0)

        st.subheader("ðŸ’° Transaction & Engagement")
        col4, col5, col6 = st.columns(3)
        with col4:
            avg_txn    = st.number_input("Avg Transaction Value (â‚¹)", 0.0, 100000.0, 25000.0)
            avg_freq   = st.slider("Avg Frequency Login Days", 0.0, 30.0, 10.0)
            points     = st.slider("Points in Wallet", 0.0, 1500.0, 700.0)
        with col5:
            used_disc  = st.selectbox("Used Special Discount", ["Yes","No"])
            offer_pref = st.selectbox("Offer Application Preference", ["Yes","No"])
            past_comp  = st.selectbox("Past Complaint", ["Yes","No"])
        with col6:
            comp_stat  = st.selectbox("Complaint Status",
                ["Not Applicable","Solved","Unsolved","Solved in Follow-up",
                 "No Information Available"])
            feedback   = st.selectbox("Feedback",
                ["Products always in Stock","Quality Customer Care","Reasonable Price",
                 "Poor Website","Poor Customer Service","Poor Product Quality",
                 "No reason specified"])
            has_ref    = st.selectbox("Has Referral?", ["No","Yes"])

        st.subheader("ðŸ“… Joining Details")
        col7, col8 = st.columns(2)
        with col7:
            joining_year  = st.selectbox("Joining Year",  list(range(2015, 2020)), index=2)
            joining_month = st.selectbox("Joining Month", list(range(1, 13)), index=7)
            joining_day   = st.selectbox("Joining Day",   list(range(1, 29)), index=14)
        with col8:
            joining_dow   = st.selectbox("Day of Week Joined (0=Mon)", list(range(7)), index=2)
            tenure        = st.slider("Customer Tenure (days)", 0, 1000, 300)
            last_hour     = st.slider("Last Visit Hour (0-23)", 0, 23, 14)

        visit_period_map = {5:"morning", 12:"afternoon", 17:"evening", 0:"night"}

        def get_period(h):
            if 5  <= h < 12: return "morning"
            if 12 <= h < 17: return "afternoon"
            if 17 <= h < 21: return "evening"
            return "night"

        through_ref = st.selectbox("Joined Through Referral", ["Yes","No"])

        submitted = st.form_submit_button("ðŸ”® Predict Churn Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "age":                        age,
            "gender":                     gender,
            "region_category":            region,
            "membership_category":        membership,
            "joined_through_referral":    through_ref,
            "preferred_offer_types":      offer_type,
            "medium_of_operation":        medium,
            "internet_option":            internet,
            "days_since_last_login":      days_login,
            "avg_time_spent":             avg_time,
            "avg_transaction_value":      avg_txn,
            "avg_frequency_login_days":   avg_freq,
            "points_in_wallet":           points,
            "used_special_discount":      used_disc,
            "offer_application_preference": offer_pref,
            "past_complaint":             past_comp,
            "complaint_status":           comp_stat,
            "feedback":                   feedback,
            "has_referral_id":            1 if has_ref == "Yes" else 0,
            "joining_year":               joining_year,
            "joining_month":              joining_month,
            "joining_day":                joining_day,
            "joining_dayofweek":          joining_dow,
            "customer_tenure_days":       tenure,
            "last_visit_hour":            last_hour,
            "visit_period":               get_period(last_hour),
        }])

        try:
            pred  = model.predict(input_data)[0]
            prob  = model.predict_proba(input_data)[0, 1]

            st.markdown("---")
            st.subheader("ðŸŽ¯ Prediction Result")

            col_a, col_b, col_c = st.columns([1,2,1])
            with col_b:
                if pred == 1:
                    st.markdown(f"""
                    <div class="pred-churn">
                      <h2 style="color:#f87171 !important; margin:0">âš ï¸ High Churn Risk</h2>
                      <p style="font-size:2.5rem; font-weight:800; color:#fca5a5 !important; margin:0.5rem 0">
                        {prob*100:.1f}%
                      </p>
                      <p style="color:#fca5a5 !important">Churn Probability</p>
                      <p style="color:#9ca3af !important; font-size:0.85rem">
                        This customer is likely to churn. Consider proactive retention strategies.
                      </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-no-churn">
                      <h2 style="color:#34d399 !important; margin:0">âœ… Low Churn Risk</h2>
                      <p style="font-size:2.5rem; font-weight:800; color:#6ee7b7 !important; margin:0.5rem 0">
                        {prob*100:.1f}%
                      </p>
                      <p style="color:#6ee7b7 !important">Churn Probability</p>
                      <p style="color:#9ca3af !important; font-size:0.85rem">
                        This customer is likely to stay. Keep up engagement activities.
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            col_x, col_y = st.columns(2)
            with col_x:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": "#e0e0e0", "size": 40}},
                    title={"text": "Churn Probability", "font": {"color": "#c4b5fd", "size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#e0e0e0"},
                        "bar":  {"color": "#ef4444" if pred == 1 else "#6366f1"},
                        "steps": [
                            {"range": [0,  40],  "color": "rgba(52,211,153,0.2)"},
                            {"range": [40, 70],  "color": "rgba(251,191,36,0.2)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.2)"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2},
                                      "thickness": 0.75, "value": 50},
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0", height=300,
                    margin=dict(t=80, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_y:
                st.subheader("ðŸ’¡ Retention Suggestions")
                if pred == 1:
                    suggestions = [
                        "ðŸ“¬ Send a personalized retention offer",
                        "ðŸŽ Upgrade membership category",
                        "ðŸ“ž Reach out via customer success team",
                        "ðŸ’¸ Offer loyalty cashback or discount",
                        "ðŸ”” Re-engage with targeted email campaigns",
                    ]
                else:
                    suggestions = [
                        "â­ Enroll in loyalty rewards program",
                        "ðŸ¤ Request referral â€” they're engaged!",
                        "ðŸŽ¯ Offer premium membership upgrade",
                        "ðŸ“Š Monitor engagement metrics monthly",
                        "ðŸ’Œ Send quarterly satisfaction surveys",
                    ]
                for s in suggestions:
                    st.markdown(f"- {s}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.warning("Make sure the model is trained. Navigate to 'ðŸ¤– Model Comparison' first.")
