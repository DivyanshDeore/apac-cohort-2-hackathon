"""
AI-Powered Decision Intelligence Platform - Single File Version

Run:
    pip install streamlit pandas numpy matplotlib
    streamlit run decision_intelligence_platform_single_file.py

This single file includes:
- synthetic civic data generation
- decision readiness scoring
- anomaly detection
- forecasting
- natural-language question answering
- recommendations
- workflow planning
- Streamlit UI

No external APIs. No LLM calls. Lightweight and GitHub-friendly.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st


ZONES = ["North", "South", "East", "West", "Central"]

METRIC_LABELS = {
    "mobility_index": "Mobility Index",
    "avg_commute_minutes": "Average Commute Minutes",
    "traffic_incidents": "Traffic Incidents",
    "emergency_calls": "Emergency Calls",
    "clinic_wait_minutes": "Clinic Wait Minutes",
    "air_quality_index": "Air Quality Index",
    "water_usage_kl": "Water Usage",
    "energy_usage_mwh": "Energy Usage",
    "waste_collected_tons": "Waste Collected",
    "recycling_rate": "Recycling Rate",
    "school_attendance_rate": "School Attendance Rate",
    "service_requests": "Service Requests",
    "citizen_sentiment": "Citizen Sentiment",
}

CORE_METRICS = list(METRIC_LABELS)

NEGATIVE_WHEN_HIGH = {
    "avg_commute_minutes",
    "traffic_incidents",
    "emergency_calls",
    "clinic_wait_minutes",
    "air_quality_index",
    "water_usage_kl",
    "energy_usage_mwh",
    "waste_collected_tons",
    "service_requests",
}

CATEGORY_TERMS = {
    "transport": "avg_commute_minutes",
    "traffic": "avg_commute_minutes",
    "commute": "avg_commute_minutes",
    "safety": "emergency_calls",
    "emergency": "emergency_calls",
    "health": "clinic_wait_minutes",
    "clinic": "clinic_wait_minutes",
    "air": "air_quality_index",
    "pollution": "air_quality_index",
    "environment": "air_quality_index",
    "water": "water_usage_kl",
    "energy": "energy_usage_mwh",
    "power": "energy_usage_mwh",
    "waste": "waste_collected_tons",
    "recycling": "recycling_rate",
    "school": "school_attendance_rate",
    "education": "school_attendance_rate",
    "service": "service_requests",
    "sentiment": "citizen_sentiment",
    "feedback": "citizen_sentiment",
}


@st.cache_data(show_spinner=False)
def generate_data(days: int = 180, feedback_records: int = 420, seed: int = 21):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    rows = []
    zone_offsets = {"North": 0.08, "South": -0.04, "East": 0.02, "West": -0.01, "Central": 0.12}

    for day_index, day in enumerate(dates):
        weekly = np.sin(2 * np.pi * day_index / 7)
        seasonal = np.sin(2 * np.pi * day_index / 45)
        heat_wave = 1 if days - 50 < day_index < days - 36 else 0
        festival_peak = 1 if days - 24 < day_index < days - 18 else 0

        for zone in ZONES:
            offset = zone_offsets[zone]
            rows.append(
                {
                    "date": day,
                    "zone": zone,
                    "mobility_index": round(float(np.clip(72 + 10 * weekly + 8 * offset + rng.normal(0, 3), 20, 100)), 2),
                    "avg_commute_minutes": round(float(np.clip(31 + 5 * weekly + 16 * offset + festival_peak * 7 + rng.normal(0, 2.2), 12, 80)), 2),
                    "traffic_incidents": int(max(0, rng.poisson(4.2 + 2.5 * offset + festival_peak * 2.5))),
                    "emergency_calls": int(max(0, rng.poisson(7.5 + 2.0 * offset + heat_wave * 2.8))),
                    "clinic_wait_minutes": round(float(np.clip(38 + 9 * offset + heat_wave * 11 + rng.normal(0, 4), 8, 95)), 2),
                    "air_quality_index": round(float(np.clip(74 + 12 * seasonal + 18 * offset + heat_wave * 24 + rng.normal(0, 6), 20, 180)), 2),
                    "water_usage_kl": round(float(np.clip(42 + 3 * seasonal + heat_wave * 8 + rng.normal(0, 2.5), 20, 80)), 2),
                    "energy_usage_mwh": round(float(np.clip(58 + 6 * seasonal + heat_wave * 10 + festival_peak * 3 + rng.normal(0, 3), 25, 105)), 2),
                    "waste_collected_tons": round(float(np.clip(18 + festival_peak * 7 + rng.normal(0, 1.8), 5, 40)), 2),
                    "recycling_rate": round(float(np.clip(46 - 4 * offset - festival_peak * 5 + rng.normal(0, 3), 10, 85)), 2),
                    "school_attendance_rate": round(float(np.clip(91 - 3 * offset - heat_wave * 2 + rng.normal(0, 1.8), 70, 99)), 2),
                    "service_requests": int(max(0, rng.poisson(15 + 5 * offset + festival_peak * 5 + heat_wave * 4))),
                    "citizen_sentiment": round(float(np.clip(0.18 - 0.06 * offset - heat_wave * 0.10 - festival_peak * 0.06 + rng.normal(0, 0.08), -1, 1)), 3),
                    "vulnerable_population_index": round(float(np.clip(0.35 + {"North": 0.10, "South": 0.18, "East": 0.12, "West": 0.08, "Central": 0.04}[zone], 0, 1)), 2),
                }
            )

    metrics = pd.DataFrame(rows)

    topics = {
        "transportation": ["bus delays near the main corridor", "unsafe pedestrian crossing by the market", "need more evening buses"],
        "public_safety": ["street lighting outage near housing blocks", "late-night safety concern at transit stop"],
        "healthcare": ["clinic wait times are too long", "mobile health camp requested for seniors"],
        "environment": ["air quality has worsened near industrial road", "park waste bins overflow"],
        "utilities": ["water pressure drops in the morning", "power interruptions during peak hours"],
        "education": ["after-school learning support requested", "digital access is hard for some students"],
    }
    feedback_rows = []
    categories = list(topics)
    feedback_dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90)

    for feedback_id in range(1, feedback_records + 1):
        category = str(rng.choice(categories, p=[0.20, 0.15, 0.15, 0.18, 0.17, 0.15]))
        zone = str(rng.choice(ZONES, p=[0.19, 0.23, 0.20, 0.18, 0.20]))
        urgency = int(rng.choice([1, 2, 3, 4, 5], p=[0.18, 0.28, 0.28, 0.18, 0.08]))
        sentiment = round(float(np.clip(rng.normal(0.05 - urgency * 0.11, 0.25), -1, 1)), 3)
        feedback_rows.append(
            {
                "feedback_id": feedback_id,
                "date": rng.choice(feedback_dates),
                "zone": zone,
                "category": category,
                "urgency": urgency,
                "sentiment": sentiment,
                "message": str(rng.choice(topics[category])),
            }
        )

    feedback = pd.DataFrame(feedback_rows).sort_values("date").reset_index(drop=True)
    return metrics, feedback


def community_score(metrics: pd.DataFrame) -> pd.DataFrame:
    scored = metrics.copy()
    parts = []
    for metric in CORE_METRICS:
        values = scored[metric].astype(float)
        low = values.quantile(0.05)
        high = values.quantile(0.95)
        normalized = ((values - low) / max(0.001, high - low)).clip(0, 1)
        if metric in NEGATIVE_WHEN_HIGH:
            normalized = 1 - normalized
        parts.append(normalized)
    scored["decision_readiness_score"] = (pd.concat(parts, axis=1).mean(axis=1) * 100).round(2)
    return scored


def latest_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["date"] == metrics["date"].max()].copy()


def kpi_cards(metrics: pd.DataFrame) -> pd.DataFrame:
    scored = community_score(metrics)
    latest = scored[scored["date"] == scored["date"].max()]
    previous = scored[scored["date"] == scored["date"].max() - pd.Timedelta(days=7)]
    if previous.empty:
        previous = scored[scored["date"] == scored["date"].min()]

    rows = []
    for metric in ["decision_readiness_score", "avg_commute_minutes", "air_quality_index", "clinic_wait_minutes", "citizen_sentiment"]:
        current = latest[metric].mean()
        old = previous[metric].mean()
        delta = current - old
        rows.append(
            {
                "metric": "Decision Readiness Score" if metric == "decision_readiness_score" else METRIC_LABELS[metric],
                "current": round(float(current), 2),
                "seven_day_change": round(float(delta), 2),
            }
        )
    return pd.DataFrame(rows)


def detect_anomalies(metrics: pd.DataFrame, threshold: float = 2.1) -> pd.DataFrame:
    rows = []
    for zone, group in metrics.groupby("zone"):
        group = group.sort_values("date")
        baseline = group.tail(60)
        latest = group.iloc[-1]
        for metric in CORE_METRICS:
            mean = baseline[metric].mean()
            std = baseline[metric].std(ddof=0)
            if not std or np.isnan(std):
                continue
            z_score = (latest[metric] - mean) / std
            if abs(z_score) >= threshold:
                rows.append(
                    {
                        "date": latest["date"].date().isoformat(),
                        "zone": zone,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "latest_value": round(float(latest[metric]), 2),
                        "baseline": round(float(mean), 2),
                        "z_score": round(float(z_score), 2),
                        "severity": "High" if abs(z_score) >= 3 else "Medium",
                        "signal": "spike" if z_score > 0 else "drop",
                    }
                )

    columns = ["date", "zone", "metric", "metric_label", "latest_value", "baseline", "z_score", "severity", "signal"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["severity", "z_score"], ascending=[True, False]).reset_index(drop=True)


def forecast_metric(metrics: pd.DataFrame, metric: str, zone: str = "All", horizon: int = 14) -> pd.DataFrame:
    data = metrics.copy()
    if zone != "All":
        data = data[data["zone"] == zone]
    daily = data.groupby("date", as_index=False)[metric].mean().sort_values("date")
    daily["x"] = np.arange(len(daily))
    recent = daily.tail(60)
    slope, intercept = np.polyfit(recent["x"].to_numpy(dtype=float), recent[metric].to_numpy(dtype=float), 1)
    seasonal = daily[metric].tail(14).mean() - daily[metric].tail(60).mean()
    last_x = int(daily["x"].max())
    last_date = daily["date"].max()

    return pd.DataFrame(
        [
            {
                "date": last_date + pd.Timedelta(days=step),
                "zone": zone,
                "metric": metric,
                "predicted_value": round(float(intercept + slope * (last_x + step) + seasonal * min(step / horizon, 1)), 2),
            }
            for step in range(1, horizon + 1)
        ]
    )


def feedback_summary(feedback: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        feedback.groupby(["category", "zone"], as_index=False)
        .agg(messages=("feedback_id", "count"), avg_urgency=("urgency", "mean"), avg_sentiment=("sentiment", "mean"))
        .sort_values(["avg_urgency", "messages"], ascending=False)
    )
    grouped["avg_urgency"] = grouped["avg_urgency"].round(2)
    grouped["avg_sentiment"] = grouped["avg_sentiment"].round(3)
    return grouped


def area_for_metric(metric: str) -> str:
    if metric in {"avg_commute_minutes", "traffic_incidents", "mobility_index"}:
        return "Transportation"
    if metric == "emergency_calls":
        return "Public Safety"
    if metric == "clinic_wait_minutes":
        return "Healthcare"
    if metric == "air_quality_index":
        return "Environment"
    if metric in {"water_usage_kl", "energy_usage_mwh"}:
        return "Utilities"
    if metric in {"waste_collected_tons", "recycling_rate"}:
        return "Waste Management"
    if metric == "school_attendance_rate":
        return "Education"
    return "Citizen Engagement"


def recommendation_for(metric: str, zone: str) -> str:
    templates = {
        "avg_commute_minutes": f"Adjust traffic signal timing and add temporary transit capacity in {zone}.",
        "traffic_incidents": f"Deploy road-safety inspection around incident hotspots in {zone}.",
        "emergency_calls": f"Pre-position emergency response resources and send safety alerts for {zone}.",
        "clinic_wait_minutes": f"Open overflow clinic slots and route non-urgent visits to telehealth for {zone}.",
        "air_quality_index": f"Trigger air-quality advisory and inspect pollution sources affecting {zone}.",
        "water_usage_kl": f"Run leak checks and send water conservation guidance in {zone}.",
        "energy_usage_mwh": f"Activate peak-load reduction and inspect smart meter anomalies in {zone}.",
        "waste_collected_tons": f"Add sanitation pickups and temporary bins in {zone}.",
        "recycling_rate": f"Launch recycling education and bin-access improvements in {zone}.",
        "school_attendance_rate": f"Coordinate school outreach and transport support for students in {zone}.",
        "service_requests": f"Create a rapid service backlog response team for {zone}.",
        "citizen_sentiment": f"Schedule a listening session and publish service recovery plan for {zone}.",
    }
    return templates.get(metric, f"Investigate unusual {metric} signal in {zone}.")


def recommend_actions(metrics: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, anomaly in detect_anomalies(metrics).head(10).iterrows():
        metric = str(anomaly["metric"])
        rows.append(
            {
                "priority": anomaly["severity"],
                "zone": anomaly["zone"],
                "decision_area": area_for_metric(metric),
                "trigger": f"{anomaly['metric_label']} {anomaly['signal']} detected",
                "recommended_action": recommendation_for(metric, str(anomaly["zone"])),
                "expected_impact": "Improves everyday service quality, resilience, and community well-being.",
                "automation": "Create ticket, assign owner, notify stakeholders, and track status.",
            }
        )

    for _, item in feedback_summary(feedback).head(8).iterrows():
        if item["avg_urgency"] >= 3.4:
            rows.append(
                {
                    "priority": "Medium",
                    "zone": item["zone"],
                    "decision_area": str(item["category"]).replace("_", " ").title(),
                    "trigger": f"{int(item['messages'])} citizen messages with urgency {item['avg_urgency']}",
                    "recommended_action": f"Open a focused response sprint for {item['category']} issues in {item['zone']}.",
                    "expected_impact": "Improves trust, response time, and service satisfaction.",
                    "automation": "Create service tickets, assign owner, and publish status update.",
                }
            )

    if not rows:
        rows.append(
            {
                "priority": "Low",
                "zone": "All",
                "decision_area": "Citizen Engagement",
                "trigger": "No major anomaly detected",
                "recommended_action": "Run a listening campaign and collect targeted feedback.",
                "expected_impact": "Improves early signal detection and community participation.",
                "automation": "Schedule survey, route responses, and summarize weekly.",
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["zone", "trigger"]).reset_index(drop=True)


def workflow_plan(recommendations: pd.DataFrame) -> pd.DataFrame:
    owners = {
        "Transportation": "Mobility Team",
        "Public Safety": "Safety Operations",
        "Healthcare": "Community Health Office",
        "Environment": "Sustainability Team",
        "Utilities": "Utility Operations",
        "Waste Management": "Sanitation Team",
        "Education": "Learning Programs Office",
        "Citizen Engagement": "Public Services Desk",
    }
    rows = []
    for index, item in recommendations.head(12).iterrows():
        rows.append(
            {
                "workflow_id": f"WF-{index + 1:03d}",
                "priority": item["priority"],
                "owner": owners.get(item["decision_area"], "City Coordination Office"),
                "zone": item["zone"],
                "task": item["recommended_action"],
                "next_step": item["automation"],
                "due_in_days": 2 if item["priority"] == "High" else 5 if item["priority"] == "Medium" else 10,
                "status": "Ready to assign",
            }
        )
    return pd.DataFrame(rows)


def extract_zone(text: str) -> str | None:
    for zone in ZONES:
        if re.search(rf"\b{zone.lower()}\b", text.lower()):
            return zone
    return None


def extract_metric(text: str) -> str | None:
    lowered = text.lower()
    for term, metric in CATEGORY_TERMS.items():
        if term in lowered:
            return metric
    for metric, label in METRIC_LABELS.items():
        if metric.replace("_", " ") in lowered or label.lower() in lowered:
            return metric
    return None


def answer_question(question: str, metrics: pd.DataFrame, feedback: pd.DataFrame) -> Dict[str, object]:
    text = question.lower().strip()
    zone = extract_zone(text)
    metric = extract_metric(text)

    if any(word in text for word in ["forecast", "predict", "next", "future"]):
        metric = metric or "air_quality_index"
        match = re.search(r"(\d+)\s*days?", text)
        horizon = max(3, min(60, int(match.group(1)))) if match else 14
        forecast = forecast_metric(metrics, metric, zone or "All", horizon)
        delta = forecast["predicted_value"].iloc[-1] - forecast["predicted_value"].iloc[0]
        direction = "rising" if delta > 1 else "declining" if delta < -1 else "stable"
        return {
            "answer": f"{METRIC_LABELS[metric]} is forecast to be {direction} for {zone or 'all zones'} over the next {horizon} days.",
            "evidence": forecast,
        }

    if any(word in text for word in ["anomaly", "unusual", "spike", "drop", "risk", "problem"]):
        anomalies = detect_anomalies(metrics)
        if zone:
            anomalies = anomalies[anomalies["zone"] == zone]
        if metric:
            anomalies = anomalies[anomalies["metric"] == metric]
        if anomalies.empty:
            return {"answer": "No major anomaly was detected for that scope.", "evidence": anomalies}
        top = anomalies.iloc[0]
        return {
            "answer": f"Strongest signal: {top['metric_label']} {top['signal']} in {top['zone']} with z-score {top['z_score']}.",
            "evidence": anomalies.head(8),
        }

    if any(word in text for word in ["recommend", "should", "action", "decision", "improve", "plan"]):
        recs = recommend_actions(metrics, feedback)
        if zone:
            recs = recs[recs["zone"] == zone]
        if recs.empty:
            recs = recommend_actions(metrics, feedback).head(3)
        top = recs.iloc[0]
        return {
            "answer": f"Recommended decision: {top['recommended_action']} Priority: {top['priority']}.",
            "evidence": recs.head(8),
        }

    if any(word in text for word in ["feedback", "citizen", "complaint", "message", "public"]):
        summary = feedback_summary(feedback)
        if zone:
            summary = summary[summary["zone"] == zone]
        top = summary.iloc[0]
        return {
            "answer": f"Citizen feedback is most urgent around {top['category']} in {top['zone']} with average urgency {top['avg_urgency']}.",
            "evidence": summary.head(10),
        }

    scored = community_score(metrics)
    latest = latest_rows(scored)

    if any(word in text for word in ["best", "worst", "highest", "lowest"]):
        metric = metric or "decision_readiness_score"
        ascending = any(word in text for word in ["worst", "lowest"])
        ranked = latest.sort_values(metric, ascending=ascending)
        top = ranked.iloc[0]
        label = "Decision Readiness Score" if metric == "decision_readiness_score" else METRIC_LABELS[metric]
        return {
            "answer": f"{top['zone']} has the {'lowest' if ascending else 'highest'} {label}: {top[metric]:.2f}.",
            "evidence": ranked[["date", "zone", metric]].head(5),
        }

    cards = kpi_cards(metrics)
    anomalies = detect_anomalies(metrics)
    recs = recommend_actions(metrics, feedback)
    return {
        "answer": f"Current decision readiness is {cards.iloc[0]['current']:.2f}. There are {len(anomalies)} anomaly signals and {len(recs)} recommended actions.",
        "evidence": cards,
    }


def sample_questions() -> List[str]:
    return [
        "What is the community status today?",
        "Which zone has the worst air quality?",
        "Show unusual safety risks",
        "Forecast clinic wait times for the next 14 days",
        "What should we do to improve transportation?",
        "Where are citizen complaints most urgent?",
    ]


def page_dashboard(metrics: pd.DataFrame, feedback: pd.DataFrame) -> None:
    st.title("AI-Powered Decision Intelligence Platform")
    cards = kpi_cards(metrics)
    cols = st.columns(len(cards))
    for col, row in zip(cols, cards.to_dict("records")):
        col.metric(row["metric"], row["current"], row["seven_day_change"])

    scored = community_score(metrics)
    daily_score = scored.groupby("date", as_index=False)["decision_readiness_score"].mean()
    zone_score = latest_rows(scored).sort_values("decision_readiness_score", ascending=False)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Decision Readiness Trend")
        st.line_chart(daily_score.set_index("date"), height=340)
    with right:
        st.subheader("Latest Zone Scores")
        st.dataframe(zone_score[["zone", "decision_readiness_score", "citizen_sentiment", "service_requests"]], hide_index=True, use_container_width=True)

    st.subheader("Top Recommended Actions")
    st.dataframe(recommend_actions(metrics, feedback).head(8), hide_index=True, use_container_width=True)


def page_ask(metrics: pd.DataFrame, feedback: pd.DataFrame) -> None:
    st.title("Ask the Platform")
    selected = st.selectbox("Try a question", sample_questions())
    question = st.text_input("Question", value=selected)
    response = answer_question(question, metrics, feedback)
    st.info(response["answer"])
    evidence = response["evidence"]
    if isinstance(evidence, pd.DataFrame) and not evidence.empty:
        st.dataframe(evidence, hide_index=True, use_container_width=True)


def page_anomalies(metrics: pd.DataFrame) -> None:
    st.title("Pattern and Anomaly Detection")
    threshold = st.slider("Sensitivity", 1.5, 3.5, 2.1, 0.1)
    anomalies = detect_anomalies(metrics, threshold)
    cols = st.columns(3)
    cols[0].metric("Signals", len(anomalies))
    cols[1].metric("High priority", int((anomalies["severity"] == "High").sum()) if not anomalies.empty else 0)
    cols[2].metric("Zones affected", anomalies["zone"].nunique() if not anomalies.empty else 0)
    if anomalies.empty:
        st.success("No major anomaly detected at this sensitivity.")
        return
    st.dataframe(anomalies, hide_index=True, use_container_width=True)
    chart = anomalies["metric_label"].value_counts().rename_axis("metric").reset_index(name="count")
    st.bar_chart(chart.set_index("metric"), height=300)


def page_forecasting(metrics: pd.DataFrame) -> None:
    st.title("Predictive Forecasting")
    metric = st.selectbox("Metric", CORE_METRICS, format_func=lambda value: METRIC_LABELS[value])
    zone = st.selectbox("Zone", ["All"] + ZONES)
    horizon = st.slider("Forecast days", 7, 60, 14)
    forecast = forecast_metric(metrics, metric, zone, horizon)
    actual = metrics if zone == "All" else metrics[metrics["zone"] == zone]
    actual = actual.groupby("date", as_index=False)[metric].mean().tail(60)
    combined = pd.concat(
        [
            actual.rename(columns={metric: "Actual"}).set_index("date")[["Actual"]],
            forecast.rename(columns={"predicted_value": "Forecast"}).set_index("date")[["Forecast"]],
        ],
        axis=1,
    )
    st.line_chart(combined, height=360)
    st.dataframe(forecast, hide_index=True, use_container_width=True)


def page_recommendations(metrics: pd.DataFrame, feedback: pd.DataFrame) -> None:
    st.title("Recommendations and Workflows")
    recs = recommend_actions(metrics, feedback)
    workflows = workflow_plan(recs)
    priority = st.multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    st.subheader("Decision Recommendations")
    st.dataframe(recs[recs["priority"].isin(priority)], hide_index=True, use_container_width=True)
    st.subheader("Automation-Ready Workflow Plan")
    st.dataframe(workflows, hide_index=True, use_container_width=True)


def page_feedback(feedback: pd.DataFrame) -> None:
    st.title("Citizen Feedback Intelligence")
    cols = st.columns(3)
    cols[0].metric("Messages", len(feedback))
    cols[1].metric("Average urgency", f"{feedback['urgency'].mean():.2f}")
    cols[2].metric("Average sentiment", f"{feedback['sentiment'].mean():.2f}")
    topic_counts = feedback["category"].value_counts().rename_axis("category").reset_index(name="messages")
    st.bar_chart(topic_counts.set_index("category"), height=300)
    st.subheader("Feedback Hotspots")
    st.dataframe(feedback_summary(feedback).head(20), hide_index=True, use_container_width=True)
    st.subheader("Recent Messages")
    st.dataframe(feedback.sort_values("date", ascending=False).head(30), hide_index=True, use_container_width=True)


def page_cloud_mapping() -> None:
    st.title("Google Cloud Deployment Map")
    st.markdown(
        """
| Platform capability | Single-file prototype | Google Cloud option |
| --- | --- | --- |
| Conversational analytics | Keyword and evidence-based assistant | Gemini on Vertex AI, ADK |
| Data warehouse | In-memory generated data | BigQuery |
| Forecasting and ML | NumPy trend forecasting | Vertex AI, BigQuery ML |
| App hosting | Streamlit | Cloud Run |
| Workflow automation | Workflow table | Cloud Functions, Workflows, Pub/Sub |
| Dashboards | Streamlit charts | Looker, Looker Studio |
| Responsible AI | Evidence tables and trigger signals | Vertex AI monitoring, Cloud Logging |
        """
    )


def main() -> None:
    st.set_page_config(page_title="Decision Intelligence Platform", layout="wide")
    metrics, feedback = generate_data()

    with st.sidebar:
        page = st.radio(
            "Page",
            [
                "Dashboard",
                "Ask the Platform",
                "Pattern Detection",
                "Forecasting",
                "Recommendations",
                "Citizen Feedback",
                "Cloud Mapping",
            ],
        )

    if page == "Dashboard":
        page_dashboard(metrics, feedback)
    elif page == "Ask the Platform":
        page_ask(metrics, feedback)
    elif page == "Pattern Detection":
        page_anomalies(metrics)
    elif page == "Forecasting":
        page_forecasting(metrics)
    elif page == "Recommendations":
        page_recommendations(metrics, feedback)
    elif page == "Citizen Feedback":
        page_feedback(feedback)
    else:
        page_cloud_mapping()


if __name__ == "__main__":
    main()
