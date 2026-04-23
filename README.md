# Asia_Pacific-supply-chain-project.

📦 Pacific Asia Supply Chain Analytics Dashboard

## 📌 Project Overview

This project is part of the AICW (Artificial Intelligence in the Contemporary World) Fellowship Program, conducted by the Edunet Foundation in collaboration with Microsoft, LinkedIn, and SAP.

The project focuses on analyzing supply chain and e-commerce order data across the Pacific Asia region using Microsoft Power BI to generate actionable business insights.

## 🎯 Objectives

- Perform data cleaning and preprocessing on raw supply chain data
Build 20+ DAX measures for KPI analysis
Develop a 9-page interactive Power BI dashboard
Identify key business problems such as late deliveries, profit imbalance, and fraud risk
Provide data-driven recommendations for supply chain optimization

## 📊 Dataset Details
- Total Records: 2,263 orders
- Fields: 48 variables
- Time Period: Dec 2017 – Dec 2018
# Regions Covered:
- Southeast Asia
- Oceania
- Eastern Asia
- South Asia

## 🧹 Data Cleaning Performed
- Standardized date formats
- Fixed 1,024 incorrect shipping dates
- Recalculated 145 discount errors
- Handled missing values
- Corrected data types
- Removed inconsistencies

## 🛠️ Tech Stack
- 📊 Microsoft Power BI
- 📈 DAX (Data Analysis Expressions)
- 📑 Microsoft Excel / Power Query
- 🤖 Power BI Features:
Key Influencers,
Smart Narrative,
Decomposition Tree

## 📈 Key KPIs
- Total Revenue: $390,818
- Net Profit: $39,502
- Profit Margin: ~10.1%
- Late Delivery Rate: 56.6% ⚠️
- Total Orders: 2,263
- Average Order Value: $172.70
## 🚨 Problem Statement

- The project identifies critical supply chain challenges:

56.6% Late Deliveries (1,281 orders)
First Class Shipping Failure: 97.5% late rate
Regional Profit Imbalance:
Oceania → $13,260
Eastern Asia → $4,031
Negative Profit Categories: Men's Clothing (-17.9%)
Fraud & Risk:
45 suspected fraud orders
48 cancelled orders

## 📊 Dashboard Structure (9 Pages)
- 1. Executive Summary
Revenue, Profit, Orders
Monthly trends
Customer segment performance
- 2. Regional Intelligence
GDP vs Profit analysis
Logistics score comparison
Late delivery funnel
- 3. Product & Category Analysis
Top-selling categories
Profit ratio analysis
Discount vs profit correlation
- 4. Shipping & Delivery
Late delivery rate (56.6%)
Shipping mode efficiency
Planned vs actual delivery time
- 5. Customer Analytics
Customer segmentation
Top cities by revenue
Payment behavior
- 6. Risk & Compliance
Fraud, cancellations, loss orders
Profit loss waterfall analysis
- 7. Root Cause Analysis
Decomposition tree to identify drivers of late delivery
- 8. Order Detail Drill-through
Transaction-level analysis
- 9. Business Insights (AI-powered)
Key Influencers visual
Smart Narrative summaries

## 🔍 Key Insights
- 📌 1. Logistics Failure
Over half of orders are delayed
First Class shipping performs the worst
- 📌 2. Regional Performance
Oceania → Most profitable
Eastern Asia → Lowest profit despite strong economy
- 📌 3. Product Insights
Consumer Electronics → Highest profit ratio (24.3%)
Men’s Clothing → Loss-making category
- 📌 4. Discount Impact
Higher discounts → lower profitability
Strong negative correlation
- 📌 5. Risk Factors
Fraud + cancellations directly reduce profit
Late delivery impacts customer satisfaction

## 💡 Business Recommendations
- 🚚 Shipping Optimization
Fix First Class shipping system immediately
Promote Standard Class & Same Day delivery

- 🌍 Regional Strategy
Expand premium offerings in Oceania
Improve logistics in South Asia
Cost optimization in Eastern Asia

- 🛍️ Product Strategy
Scale high-margin categories
Limit discounts (max 10–12%)
Eliminate loss-making products

- 🔐 Risk Management
Implement fraud detection alerts
Monitor cancellations
Improve payment follow-ups

- 🤖 AI Integration
Key Influencers: Identifies drivers of late delivery
Smart Narrative: Auto-generates insights
Decomposition Tree: Root cause analysis
📽️ Project Demo

🔗 Watch Demo Video

📁 Project Files
📊 Power BI Dashboard (.pbix)
📑 Project Report (.docx)
📽️ Presentation (.pptx)
⚠️ Limitations
Static dataset (no real-time updates)
Limited to 1-year data
No predictive analytics models
🚀 Future Scope
Real-time data integration
Machine learning prediction models
Expansion to full APAC region
Power Apps integration
Reusable BI templates
👩‍💻 Author

Ishita Singhal
BBA – International Business Management

🙌 Acknowledgment

Special thanks to:

Edunet Foundation
Microsoft
LinkedIn
SAP

for providing the platform and resources for this project.

---

## 🖥️ Streamlit App – Power BI Risk Console

This repository also includes a full-featured **Streamlit web application** (`app.py`) that wraps the embedded Power BI dashboard and adds live barcode scanning and ML-powered order risk detection.

### ✨ Features

#### 📊 Tab 1 – Dashboard Viewer
- Embeds the 9-page Power BI report directly inside the Streamlit UI
- Sidebar navigation buttons let you switch between Power BI report pages without leaving the app
- Supports both the Power BI JavaScript SDK (with authentication) and a simple iframe fallback

#### 🔍 Tab 2 – Barcode / Order Lookup
- **Image upload mode**: upload a screenshot or photo containing a barcode or QR code
- **Live camera mode**: uses the device camera via `streamlit-webrtc` (or falls back to `st.camera_input`)
- Multi-decoder pipeline tries **zxing-cpp → OpenCV → pyzbar → QR fallback** for best decode accuracy
- Crop inspection and retry decode on individual detected regions
- Match the decoded barcode value against an uploaded Excel or CSV order sheet (auto-detects barcode, SKU, product code, or order ID columns)
- Scan history and saveable lookup logs

#### 🚨 Tab 3 – Order Risk & Anomaly Detector
- Upload any Excel or CSV order sheet and automatically detect suspicious rows
- Combines **Isolation Forest** (scikit-learn) with statistical z-score and IQR outlier signals
- Detects amount mismatches (qty × price ≠ total), price/quantity deviations within product groups, and rare categorical values
- Configurable sensitivity, date range filter, and item/barcode text filter
- Outputs a ranked risk table with per-row explanations, a review recommendation box, and an interactive Altair scatter chart
- Barcode-linked risk card shows the anomaly score for the currently scanned order

### 🛠️ App Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Streamlit |
| Power BI embedding | iframe / Power BI JS SDK |
| Data processing | Pandas, NumPy |
| Anomaly detection | scikit-learn (Isolation Forest, StandardScaler) |
| Barcode decoding | zxing-cpp, OpenCV, pyzbar, RapidOCR |
| Live camera | streamlit-webrtc, av |
| Visualisation | Altair |
| Vision-language models | 🤗 Transformers (FastVLM 0.5B, Qwen2.5-VL 3B), PyTorch |
| Object detection | Ultralytics (YOLOv8) |
| File formats | openpyxl (Excel), Pillow (images) |

### 🚀 Running the App

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the app**
   ```bash
   streamlit run app.py
   ```

3. **(Optional) Configure Power BI credentials**

   Create a `.streamlit/secrets.toml` file (or set environment variables) with your Power BI connection details:
   ```toml
   POWERBI_IFRAME_URL = "https://app.powerbi.com/reportEmbed?reportId=..."
   POWERBI_REPORT_ID  = "your-report-id"
   POWERBI_TENANT_ID  = "your-tenant-id"
   POWERBI_GROUP_ID   = "your-workspace-id"
   POWERBI_CLIENT_ID  = "your-client-id"
   POWERBI_CLIENT_SECRET = "your-client-secret"
   ```
   Without credentials the app uses iframe mode and still embeds the public report URL bundled with the project.

### 📁 App File Structure

```
app.py                  # Main Streamlit application
requirements.txt        # Python dependencies
utils/
  barcode_utils.py      # Barcode detection, decoding, and order lookup
  powerbi_utils.py      # Power BI page discovery and embed HTML builder
  analysis_utils.py     # Anomaly feature engineering and scoring
  vlm_utils.py          # Vision-language model helpers (FastVLM / Qwen2.5-VL)
*.pbix                  # Power BI desktop file (dashboard source)
```
