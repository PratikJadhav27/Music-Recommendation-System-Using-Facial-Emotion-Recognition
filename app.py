import hashlib
import math
import os
import time

import numpy as np
import streamlit as st
from PIL import Image

from app_theme import THEME_OPTIONS, inject_theme, render_hero_subtitle
from emotion_detector import ensemble_models_available, predict_emotion
from spotify_recommendation import get_playlist_for_emotion
from user_errors import humanize_processing_error, humanize_song_fetch_error

# Top-class confidence below this (%) is treated as unreliable for song recommendations.
DEFAULT_CONFIDENCE_THRESHOLD = 40.0

st.set_page_config(page_title="Music & Emotion", layout="wide", page_icon="🎵")

app_theme = st.sidebar.selectbox(
    "App theme",
    THEME_OPTIONS,
    index=0,
    help="Studio Dark applies a music-app style palette; Default uses native Streamlit styling.",
)
inject_theme(app_theme)

st.title("🎵 Music Recommendation System using Facial Emotion Recognition")
render_hero_subtitle()

st.sidebar.header("Input")
option = st.sidebar.radio(
    "Choose an option:",
    ("Upload an Image", "Capture via Webcam", "Live Webcam (real-time)"),
)
use_face_detection = st.sidebar.checkbox(
    "Detect & crop face (recommended)",
    value=True,
    help="Uses OpenCV Haar cascades to find the largest face before resizing to 48×48. "
    "Uncheck to always resize the whole image (legacy, less accurate).",
)
min_confidence_pct = st.sidebar.slider(
    "Low-confidence cutoff (%)",
    min_value=15.0,
    max_value=75.0,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=1.0,
    help="If the top emotion score is below this, song recommendations are hidden unless you opt in.",
)
_ensemble_ok = ensemble_models_available()
use_ensemble = st.sidebar.checkbox(
    "Ensemble prediction (baseline + balanced)",
    value=_ensemble_ok,
    disabled=not _ensemble_ok,
    help="Average softmax outputs from fer_model.h5 and fer_balanced.h5 for more robust labels. "
    "Requires both files in Model/.",
)
emotion_chart_style = st.sidebar.selectbox(
    "Emotion scores chart",
    ("Radar", "Bar", "Both"),
    help="Radar shows all seven emotions on one plot; bar is Streamlit's default chart.",
)


def capture_webcam():
    """Single snapshot from the browser camera (Streamlit)."""
    from PIL import UnidentifiedImageError

    os.makedirs("images", exist_ok=True)
    picture = st.camera_input("Take a picture")
    if picture is not None:
        try:
            img = Image.open(picture)
            img.load()
            img.save("images/captured_image.jpg")
            return img
        except UnidentifiedImageError as e:
            summary, tech = humanize_processing_error(e)
            st.error(f"❌ {summary}")
            with st.expander("Technical details"):
                st.code(tech)
        except OSError as e:
            summary, tech = humanize_processing_error(e)
            st.error(f"❌ {summary}")
            with st.expander("Technical details"):
                st.code(tech)
    return None


def render_emotion_readout(
    emotion: str,
    confidence: float,
    confidence_scores: dict,
    chart_style: str = "Radar",
):
    st.subheader(f"🎭 Detected Emotion: **{emotion.capitalize()}** ({confidence:.2f}% confidence)")

    if chart_style in ("Radar", "Both"):
        from emotion_charts import plot_emotion_radar

        fig = plot_emotion_radar(confidence_scores, highlight=emotion)
        st.pyplot(fig, clear_figure=True)

    if chart_style in ("Bar", "Both"):
        if chart_style == "Both":
            st.markdown("**Bar chart**")
        st.bar_chart(confidence_scores)


def _song_refresh_version(key_prefix: str) -> int:
    return int(st.session_state.get(f"{key_prefix}_song_refresh", 0))


def render_song_recommendations(emotion: str, confidence_scores: dict, key_prefix: str = "main"):
    st.subheader("🎵 Recommended Songs for You:")
    rv = _song_refresh_version(key_prefix)
    list_key = f"{key_prefix}_song_list_{rv}"

    c_refresh, c_per_page = st.columns([1, 2])
    with c_refresh:
        if st.button(
            "🔄 New songs",
            key=f"{key_prefix}_new_songs_{rv}",
            help="Fetch a different set of songs for the same emotion (iTunes search is randomized).",
        ):
            st.session_state[f"{key_prefix}_song_refresh"] = rv + 1
            st.session_state.pop(f"{key_prefix}_song_page", None)
            st.rerun()
    with c_per_page:
        per_page = st.selectbox(
            "Songs per page",
            options=[3, 5, 8],
            index=1,
            key=f"{key_prefix}_songs_per_page_{rv}",
        )

    if list_key not in st.session_state:
        with st.spinner("🔄 Fetching songs..."):
            try:
                st.session_state[list_key] = get_playlist_for_emotion(emotion, confidence_scores)
                st.session_state[f"{key_prefix}_song_page"] = 0
            except Exception as e:
                summary, tech = humanize_song_fetch_error(e)
                st.error(f"❌ {summary}")
                with st.expander("Technical details"):
                    st.code(tech)
                return

    tracks = st.session_state.get(list_key, [])
    if not tracks:
        st.warning("⚠️ No songs found. Please check your internet connection and try again.")
        return

    total = len(tracks)
    total_pages = max(1, math.ceil(total / per_page))
    page = int(st.session_state.get(f"{key_prefix}_song_page", 0))
    page = max(0, min(page, total_pages - 1))
    st.session_state[f"{key_prefix}_song_page"] = page

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("← Previous", key=f"{key_prefix}_prev_{rv}_{page}", disabled=page <= 0):
            st.session_state[f"{key_prefix}_song_page"] = page - 1
            st.rerun()
    with nav2:
        st.caption(f"Page **{page + 1}** of **{total_pages}** · **{total}** songs total")
    with nav3:
        if st.button("Next →", key=f"{key_prefix}_next_{rv}_{page}", disabled=page >= total_pages - 1):
            st.session_state[f"{key_prefix}_song_page"] = page + 1
            st.rerun()

    start = page * per_page
    page_tracks = tracks[start : start + per_page]

    for idx, track in enumerate(page_tracks):
        global_idx = start + idx
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            if track["image"]:
                st.image(track["image"], width=100)
        with col2:
            st.markdown(f"**[{track['name']}]({track['url']})**")
            if track.get("preview"):
                st.audio(track["preview"], format="audio/mp4")
        with col3:
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("👍", key=f"{key_prefix}_like_{global_idx}_{emotion}_{rv}"):
                    from feedback_manager import log_feedback

                    if log_feedback(
                        emotion=emotion,
                        confidence_scores=confidence_scores,
                        song_name=track["name"],
                        song_url=track["url"],
                        rating=1,
                    ):
                        st.success("Thanks for the feedback!", icon="✅")
            with fc2:
                if st.button("👎", key=f"{key_prefix}_dislike_{global_idx}_{emotion}_{rv}"):
                    from feedback_manager import log_feedback

                    if log_feedback(
                        emotion=emotion,
                        confidence_scores=confidence_scores,
                        song_name=track["name"],
                        song_url=track["url"],
                        rating=-1,
                    ):
                        st.info("Feedback noted!", icon="ℹ️")


def maybe_render_songs_after_emotion(
    emotion: str,
    confidence: float,
    confidence_scores: dict,
    key_prefix: str,
    threshold: float,
    chart_style: str = "Radar",
):
    """Show songs unless confidence is below threshold (then require opt-in)."""
    render_emotion_readout(emotion, confidence, confidence_scores, chart_style=chart_style)
    if confidence < threshold:
        st.warning(
            f"**Low confidence** ({confidence:.1f}% is below {threshold:.0f}%). "
            "The prediction may not match how you feel — try a clearer, front-facing photo or better lighting."
        )
        if st.checkbox("Show song recommendations anyway", key=f"{key_prefix}_override_low_conf"):
            render_song_recommendations(emotion, confidence_scores, key_prefix=key_prefix)
    else:
        render_song_recommendations(emotion, confidence_scores, key_prefix=key_prefix)

def render_gradcam(image: Image.Image, model_input: np.ndarray, class_index: int | None = None):
    st.subheader("🔍 Explainability (Grad-CAM)")
    show = st.checkbox("Show Grad-CAM heatmap", value=False)
    if not show:
        return

    alpha = st.slider("Heatmap intensity", min_value=0.0, max_value=0.9, value=0.45, step=0.05)
    try:
        with st.spinner("Generating Grad-CAM heatmap..."):
            from gradcam import compute_gradcam_heatmap, overlay_heatmap_on_image

            heatmap = compute_gradcam_heatmap(model_input, class_index=class_index)
            overlay = overlay_heatmap_on_image(image, heatmap, alpha=alpha)

        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original", use_container_width=True)
        with c2:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
    except Exception as e:
        summary, tech = humanize_processing_error(e)
        st.warning(f"Grad-CAM: {summary}")
        with st.expander("Grad-CAM technical details"):
            st.code(tech)


def pil_rgb_fingerprint(img: Image.Image) -> str:
    return hashlib.sha256(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


# --- Upload / snapshot webcam ---
image = None
if option == "Upload an Image":
    uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        from PIL import UnidentifiedImageError

        try:
            image = Image.open(uploaded)
            image.load()
        except UnidentifiedImageError as e:
            summary, tech = humanize_processing_error(e)
            st.error(f"❌ {summary}")
            with st.expander("Technical details"):
                st.code(tech)
            image = None
        except OSError as e:
            summary, tech = humanize_processing_error(e)
            st.error(f"❌ {summary}")
            with st.expander("Technical details"):
                st.code(tech)
            image = None
elif option == "Capture via Webcam":
    image = capture_webcam()

if image is not None:
    if image.width < 32 or image.height < 32:
        st.info("This image is very small; emotion results may be unreliable. Try at least ~200 px on the shortest side if you can.")

    try:
        from face_preprocess import pil_to_model_input_from_face

        try:
            cache_tag = f"{option}_{use_face_detection}_{pil_rgb_fingerprint(image)}"
        except MemoryError as e:
            summary, tech = humanize_processing_error(e)
            st.error(f"❌ {summary}")
            with st.expander("Technical details"):
                st.code(tech)
            st.stop()
        except Exception as e:
            summary, tech = humanize_processing_error(e)
            st.error(f"❌ Could not fingerprint image for caching: {summary}")
            with st.expander("Technical details"):
                st.code(tech)
            st.stop()

        cached = st.session_state.get("upload_static_cache_tag") == cache_tag

        if not cached:
            st.session_state["upload_song_refresh"] = 0

        if cached:
            emotion = st.session_state["upload_static_emotion"]
            confidence = float(st.session_state["upload_static_confidence"])
            confidence_scores = dict(st.session_state["upload_static_scores"])
            batch = st.session_state["upload_static_batch"]
            display_img = st.session_state["upload_static_display"]
            gradcam_base = st.session_state.get("upload_static_gradcam")
        else:
            with st.spinner("🔄 Detecting face & analyzing emotion..."):
                batch, display_img, gradcam_base, face_err = pil_to_model_input_from_face(
                    image,
                    require_face=use_face_detection,
                )
            if batch is None:
                st.warning(face_err or "Could not prepare image.")
                st.caption(
                    "Tips: use a **front-facing** photo with your face visible, improve **lighting**, "
                    "or turn off **Detect & crop face** if the detector keeps missing you."
                )
                with st.expander("Why this can happen"):
                    st.markdown(
                        "- Haar cascades work best on frontal faces in reasonable light.\n"
                        "- Very small faces in a wide shot may not be detected.\n"
                        "- Sunglasses / heavy occlusion can hide the face region."
                    )
                st.stop()

            with st.spinner("🔄 Analyzing emotion..."):
                try:
                    emotion, confidence, confidence_scores = predict_emotion(batch, use_ensemble=use_ensemble)
                except Exception as e:
                    summary, tech = humanize_processing_error(e)
                    st.error(f"❌ Emotion model failed: {summary}")
                    with st.expander("Technical details"):
                        st.code(tech)
                    st.stop()

            st.session_state["upload_static_cache_tag"] = cache_tag
            st.session_state["upload_static_emotion"] = emotion
            st.session_state["upload_static_confidence"] = confidence
            st.session_state["upload_static_scores"] = dict(confidence_scores)
            st.session_state["upload_static_batch"] = np.array(batch, copy=True)
            st.session_state["upload_static_display"] = display_img
            st.session_state["upload_static_gradcam"] = gradcam_base

        st.image(display_img, caption="Image used for prediction (green box = face region)", use_container_width=True)
        if use_ensemble and _ensemble_ok:
            st.caption("Prediction mode: **ensemble** (baseline + balanced CNN averaged).")

        maybe_render_songs_after_emotion(
            emotion,
            confidence,
            confidence_scores,
            key_prefix="upload",
            threshold=min_confidence_pct,
            chart_style=emotion_chart_style,
        )
        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        class_index = labels.index(emotion) if emotion in labels else None
        gcam_img = gradcam_base if gradcam_base is not None else image
        render_gradcam(gcam_img, batch, class_index=class_index)
    except Exception as e:
        summary, tech = humanize_processing_error(e)
        st.error(f"❌ {summary}")
        with st.expander("Technical details"):
            st.code(tech)

# --- Live WebRTC ---
elif option == "Live Webcam (real-time)":
    from streamlit_webrtc import webrtc_streamer

    from realtime_webcam import (
        RTC_CONFIGURATION,
        live_state,
        make_video_frame_callback,
        reset_live_state,
    )

    st.markdown(
        "Live mode streams your webcam and overlays the predicted emotion on each frame. "
        "**Allow camera access** when the browser asks. "
        "On Streamlit Cloud, HTTPS is already enabled; on your PC, `http://localhost` is enough for camera access."
    )
    if st.sidebar.button("Reset live session"):
        reset_live_state()
        st.rerun()

    if use_ensemble and _ensemble_ok:
        st.caption("Live prediction mode: **ensemble** (baseline + balanced CNN averaged).")

    ctx = webrtc_streamer(
        key="emotion-live",
        video_frame_callback=make_video_frame_callback(use_ensemble=use_ensemble and _ensemble_ok),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    status = st.empty()
    chart_slot = st.empty()

    if ctx.state.playing:
        while ctx.state.playing:
            with live_state.lock:
                em = live_state.emotion
                conf = live_state.confidence
                scores = dict(live_state.scores)
                err = live_state.error
            if err:
                status.error(f"Live inference: {err}")
            elif em:
                status.markdown(f"**Live readout:** {em.capitalize()} — **{conf:.1f}%**")
                if scores:
                    chart_slot.bar_chart(scores)
            time.sleep(0.2)

        with live_state.lock:
            if live_state.emotion:
                sig = (
                    live_state.emotion,
                    round(float(live_state.confidence), 2),
                    tuple(sorted((k, round(float(v), 2)) for k, v in live_state.scores.items())),
                )
                if st.session_state.get("live_reading_sig") != sig:
                    st.session_state["live_song_refresh"] = 0
                    st.session_state["live_reading_sig"] = sig
                st.session_state["live_emotion"] = live_state.emotion
                st.session_state["live_confidence"] = live_state.confidence
                st.session_state["live_scores"] = dict(live_state.scores)

    if not ctx.state.playing and st.session_state.get("live_emotion"):
        st.divider()
        st.subheader("Songs for your last live reading")
        maybe_render_songs_after_emotion(
            st.session_state["live_emotion"],
            float(st.session_state.get("live_confidence", 0)),
            st.session_state.get("live_scores") or {},
            key_prefix="live",
            threshold=min_confidence_pct,
            chart_style=emotion_chart_style,
        )
        if st.button("Clear live results"):
            for k in (
                "live_emotion",
                "live_confidence",
                "live_scores",
                "live_reading_sig",
                "live_song_refresh",
            ):
                st.session_state.pop(k, None)
            st.rerun()
