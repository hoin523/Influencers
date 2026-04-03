"""Streamlit review dashboard. Read-only for persona info, editable for content queue status."""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def api(method: str, path: str, **kwargs):
    resp = getattr(requests, method)(f"{API_BASE}{path}", **kwargs)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="AI Influencer Factory", layout="wide")
st.title("AI Influencer Factory")

# Sidebar: persona list (read-only)
st.sidebar.header("Personas")
try:
    personas = api("get", "/personas")
except requests.ConnectionError:
    st.error("Backend not running. Start with: python main.py")
    st.stop()

if not personas:
    st.info("No personas found. Add YAML files to personas/ directory and restart the backend.")
    st.stop()

persona_names = {p["id"]: p["name"] for p in personas}
selected_id = st.sidebar.selectbox(
    "Select persona",
    options=list(persona_names.keys()),
    format_func=lambda x: persona_names[x],
)

selected = next(p for p in personas if p["id"] == selected_id)
st.sidebar.markdown(f"**{selected['name']}** ({selected['age']}, {selected['gender']})")
st.sidebar.markdown(f"Niche: {selected['niche']}")
st.sidebar.markdown(f"Style: {selected['speaking_style']}")

# Main area: content queue
tab_generate, tab_review, tab_posted = st.tabs(["Generate", "Review", "Posted"])

with tab_generate:
    st.subheader("Generate Content")
    days = st.number_input("Days", min_value=1, max_value=30, value=7)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Calendar", type="primary"):
            with st.spinner("Generating content calendar via LLM..."):
                try:
                    result = api("post", f"/personas/{selected_id}/generate", params={"days": days})
                    st.success(f"Generated {result['generated']} content items")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with col2:
        if st.button("Generate Images"):
            with st.spinner("Generating images via ComfyUI... This may take a few minutes."):
                try:
                    stats = api("post", f"/personas/{selected_id}/generate-images")
                    st.success(f"Success: {stats['success']}, Failed: {stats['failed']}")
                except Exception as e:
                    st.error(f"Failed: {e}")

with tab_review:
    st.subheader("Review & Approve")

    status_filter = st.selectbox(
        "Filter by status",
        ["generated", "planned", "error", "all"],
    )
    params = {"persona_id": selected_id}
    if status_filter != "all":
        params["status"] = status_filter

    items = api("get", "/content-queue", params=params)

    if not items:
        st.info("No content items found.")
    else:
        for item in items:
            with st.expander(
                f"[{item['status']}] {item['post_date']} — {item['concept'][:50]}",
                expanded=item["status"] == "generated",
            ):
                col_img, col_text = st.columns([1, 2])

                with col_img:
                    if item.get("image_path"):
                        st.image(item["image_path"], width=300)
                    else:
                        st.info("No image generated yet")

                with col_text:
                    st.markdown(f"**Caption:** {item['caption']}")
                    st.markdown(f"**Hashtags:** {item.get('hashtags', '[]')}")
                    st.markdown(f"**Image prompt:** {item['image_prompt']}")

                    if item.get("error_message"):
                        st.error(f"Error: {item['error_message']}")

                # Action buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)

                if item["status"] == "generated":
                    with btn_col1:
                        if st.button("Approve", key=f"approve_{item['id']}"):
                            api("patch", f"/content-queue/{item['id']}", json={"status": "approved"})
                            st.rerun()
                    with btn_col2:
                        if st.button("Reject", key=f"reject_{item['id']}"):
                            api("patch", f"/content-queue/{item['id']}", json={"status": "planned"})
                            st.rerun()

                if item["status"] == "error":
                    with btn_col1:
                        if st.button("Retry", key=f"retry_{item['id']}"):
                            api("patch", f"/content-queue/{item['id']}", json={"status": "planned"})
                            st.rerun()

with tab_posted:
    st.subheader("Approved & Posted")
    approved = api("get", "/content-queue", params={"persona_id": selected_id, "status": "approved"})
    posted = api("get", "/content-queue", params={"persona_id": selected_id, "status": "posted"})

    for item in approved + posted:
        with st.expander(f"[{item['status']}] {item['post_date']} — {item['concept'][:50]}"):
            if item.get("image_path"):
                st.image(item["image_path"], width=300)
            st.markdown(f"**Caption:** {item['caption']}")

            if item["status"] == "approved":
                if st.button("Mark as Posted", key=f"posted_{item['id']}"):
                    api("patch", f"/content-queue/{item['id']}", json={"status": "posted"})
                    st.rerun()
