# Real-Time Business Intelligence & Analytics Platform

A high-performance, live-streaming Business Intelligence (BI) dashboard that tracks transactional data in real-time and applies predictive machine learning models to forecast future revenue streams. 

The architecture is built using a decoupled approach: a continuous data simulation engine feeds a localized database, while a Streamlit frontend delivers sub-second analytical updates and predictive insights.

---

## 🚀 Key Features

* **Live Data Streaming:** Simulates concurrent global transactions with realistic bulk-purchase market spikes, streaming directly into an indexed SQLite layout.
* **Real-Time Analytics Display:** Dynamic KPI tracking including Total Revenue, Total Orders, Average Order Value (AOV), and Units Sold with automatic half-window delta calculations.
* **Advanced Interactive Visualizations:** Built-in Plotly charts capturing revenue per minute, regional market shares, product category breakdowns, and transactional scatter distributions.
* **Predictive Forecasting Engine:** Features a integrated machine learning model utilizing degree-2 polynomial feature engineering and Ridge Regression to project revenue trends for the next 60 minutes, complete with 1-σ confidence intervals.

---

## 🛠️ Tech Stack

* **Frontend Dashboard:** Streamlit, Plotly Express, HTML5/Custom CSS
* **Data Science & ML:** Python, Pandas, NumPy, Scikit-Learn (Ridge Regression)
* **Database Engine:** SQLite3 (Indexed for minutely aggregations)

---

## 📂 Repository Structure

* `app.py` - The core Streamlit application containing the UI rendering loop, custom layout styling, and automated data refresh cycles.
* `data_generator.py` - Background simulation utility that streams live relational transaction records.
* `predictor.py` - Machine learning script handling feature engineering (sin/cos temporal encodings) and predictive analytics.
* `requirements.txt` - Project dependencies specifying isolated package versions.

---

## ⚡ Getting Started

### 1. Environment Setup
It is highly recommended to isolate your dependencies using a Python virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
