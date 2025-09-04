import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# --- Page config ---
st.set_page_config(page_title="Colourful Loan Calculator", layout="wide", initial_sidebar_state="expanded")

# --- CSS for colourful background and improved UI/UX ---
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 25%, #f6a5c0 50%, #c1d3ff 75%, #d4f1be 100%);
        background-attachment: fixed;
    }
    /* Card style */
    .card {
        background: rgba(255,255,255,0.85);
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    h1 { font-weight: 800; }
    .small { font-size:0.9rem; color:#333; }
    /* Tweak input width */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        min-width: 220px;
    }
    /* Customize slider appearance slightly */
    .stSlider>div>div>input {
        min-width: 220px;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Title and description ---
st.title(" BudgetBridge ")
st.write("Use the controls to model loan scenarios, view amortization schedules, charts, and download results. Designed with accessibility and a colourful UI in mind.")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Borrower & Loan Details")
    name = st.text_input("Full name", value="John Doe")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    occupation = st.text_input("Occupation (optional)", value="Engineer")
    st.markdown("---")

    loan_type = st.radio("Loan type", ("Home Loan", "Personal Loan", "Auto Loan", "Other"))
    principal = st.number_input("Loan amount (principal)", min_value=0.0, step=1000.0, value=250000.0, format="%.2f")
    deposit = st.number_input("Deposit / Down payment", min_value=0.0, step=100.0, value=25000.0, format="%.2f")
    effective_principal = max(0.0, principal - deposit)

    interest_rate = st.slider("Annual interest rate (%)", min_value=0.0, max_value=25.0, value=7.5, step=0.1)
    compounding = st.selectbox("Compounding frequency", ("Monthly", "Quarterly", "Annually"))
    duration_years = st.number_input("Duration (years)", min_value=1, max_value=50, value=20)
    extra_payment = st.number_input("Extra monthly payment (optional)", min_value=0.0, step=50.0, value=0.0, format="%.2f")
    include_taxes = st.checkbox("Include estimated property tax (annual, 0 if N/A)", value=False)
    est_tax = 0.0
    if include_taxes:
        est_tax = st.number_input("Estimated annual property tax", min_value=0.0, value=2000.0, format="%.2f")

    show_amort = st.checkbox("Show amortization schedule", value=True)
    show_graphs = st.checkbox("Show charts", value=True)
    download_csv = st.checkbox("Enable CSV download of amortization schedule", value=True)
    st.markdown("---")
    st.header("Simulation & Options")
    inflation_adj = st.checkbox("Adjust for simple inflation (annual %)", value=False)
    inflation_rate = 2.0
    if inflation_adj:
        inflation_rate = st.slider("Inflation rate (%)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
    rounding = st.checkbox("Round currency to nearest rupee/dollar", value=True)
    st.caption("Tip: Use the inputs to the left — results update instantly.")

# --- Input validation and summary ---
st.subheader("Summary")
col1, col2, col3 = st.columns([3,2,2])
with col1:
    st.markdown(f"**Borrower:** {name} ({occupation}), Age {age}")
    st.markdown(f"**Loan type:** {loan_type}")
with col2:
    st.markdown(f"**Loan amount:** {principal:,.2f}")
    st.markdown(f"**Deposit:** {deposit:,.2f}")
    st.markdown(f"**Principal financed:** {effective_principal:,.2f}")
with col3:
    st.markdown(f"**Interest rate:** {interest_rate:.2f}% per annum")
    st.markdown(f"**Duration:** {duration_years} years")
    st.markdown(f"**Extra monthly:** {extra_payment:,.2f}")

if effective_principal <= 0:
    st.error("Deposit covers the loan amount. Adjust deposit or principal.")
    st.stop()

# --- Calculations ---
# Convert interest to monthly rate
freq_map = {"Monthly":12, "Quarterly":4, "Annually":1}
periods_per_year = freq_map.get(compounding, 12)
total_months = int(duration_years * 12)

# Monthly interest approximation from annual and compounding
annual_rate = float(interest_rate) / 100.0
if compounding == "Monthly":
    monthly_rate = (1 + annual_rate) ** (1/12) - 1
elif compounding == "Quarterly":
    monthly_rate = (1 + annual_rate) ** (1/4) ** (1/3) - 1  # approximate to monthly
else:
    monthly_rate = (1 + annual_rate) ** (1/12) - 1

# Edge case: zero interest
def monthly_payment(P, r, n):
    if n == 0:
        return 0.0
    if r == 0:
        return P / n
    return P * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

monthly_rate_float = monthly_rate
base_monthly_payment = monthly_payment(effective_principal, monthly_rate_float, total_months)
monthly_payment_with_extra = base_monthly_payment + extra_payment

# Build amortization schedule
schedule_rows = []
balance = effective_principal
total_interest_paid = 0.0
total_principal_paid = 0.0
for m in range(1, total_months + 1):
    interest = balance * monthly_rate_float
    principal_paid = min(balance, monthly_payment_with_extra - interest)
    if monthly_rate_float == 0:
        # evenly pay principal if 0 interest
        principal_paid = min(balance, monthly_payment_with_extra)
        interest = 0.0
    balance = balance - principal_paid
    total_interest_paid += interest
    total_principal_paid += principal_paid
    schedule_rows.append({
        "Month": m,
        "Payment": monthly_payment_with_extra if balance>0 else principal_paid + interest,
        "Principal": principal_paid,
        "Interest": interest,
        "Balance": max(0.0, balance)
    })
    if balance <= 0.0001:
        break

amort_df = pd.DataFrame(schedule_rows)
amort_df['Cumulative Interest'] = amort_df['Interest'].cumsum()
amort_df['Cumulative Principal'] = amort_df['Principal'].cumsum()

# Adjust for inflation if selected (simple)
if inflation_adj and inflation_rate > 0:
    inflation_factor = [(1 + inflation_rate/100)**(i/12) for i in amort_df['Month']]
    amort_df['Balance (inflation adj)'] = amort_df['Balance'] / inflation_factor

# Rounding display
if rounding:
    display_df = amort_df.round(2)
else:
    display_df = amort_df.copy()

# --- Results ---
st.subheader("Results & Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Monthly payment (base)", f"{base_monthly_payment:,.2f}")
    st.metric("Monthly payment (with extra)", f"{monthly_payment_with_extra:,.2f}")
with col2:
    st.metric("Total interest (so far)", f"{total_interest_paid:,.2f}")
    st.metric("Total principal paid", f"{total_principal_paid:,.2f}")
with col3:
    st.metric("Months to finish (approx)", f"{len(amort_df)}")

# --- Download CSV ---
if download_csv and not amort_df.empty:
    csv = amort_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download amortization CSV", data=csv, file_name="amortization_schedule.csv", mime="text/csv")

# --- Show amortization schedule ---
if show_amort:
    st.subheader("Amortization Schedule")
    st.write("You can sort and filter the table below.")
    st.dataframe(display_df, height=320)

# --- Charts ---
if show_graphs:
    st.subheader("Charts")

    # 1) Balance over time (matplotlib)
    fig1, ax1 = plt.subplots(figsize=(8,3.5))
    ax1.plot(amort_df['Month'], amort_df['Balance'])
    ax1.set_title("Loan Balance Over Time")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Remaining Balance")
    ax1.grid(True)
    st.pyplot(fig1)

    # 2) Payment components stacked area (principal vs interest)
    fig2, ax2 = plt.subplots(figsize=(8,3.5))
    ax2.stackplot(amort_df['Month'], amort_df['Principal'], amort_df['Interest'], labels=['Principal','Interest'])
    ax2.set_title("Principal vs Interest Over Time (stacked)")
    ax2.set_xlabel("Month")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # 3) Pie showing final breakdown (principal vs interest)
    try:
        fig3, ax3 = plt.subplots(figsize=(4,4))
        principal_total = amort_df['Principal'].sum()
        interest_total = amort_df['Interest'].sum()
        ax3.pie([principal_total, interest_total], labels=['Principal','Interest'], autopct='%1.1f%%', startangle=90)
        ax3.set_title("Principal vs Interest (overall)")
        st.pyplot(fig3)
    except Exception:
        st.write("Cannot render pie chart in this environment.")

# --- Simple sensitivity analysis ---
st.subheader("Quick Sensitivity: Compare different interest rates")
rates = [max(0, interest_rate - 2), interest_rate, interest_rate + 2]
comp_cols = st.columns(3)
for i, r in enumerate(rates):
    months = total_months
    r_month = (1 + r/100) ** (1/12) - 1
    pay = monthly_payment(effective_principal, r_month, months)
    comp_cols[i].metric(f"Rate = {r:.1f}%", f"{pay:,.2f}")

# --- Footer ---
st.caption("Built with ❤️ — tweak inputs on the left. Export and share scenarios.")

