from pathlib import Path
import streamlit as st
import re

def get_slide_paths(slides_dir: str) -> list[Path]:
    slides_path = Path(slides_dir)

    slides = (
        list(slides_path.glob("Slide*.PNG")) +
        list(slides_path.glob("Slide*.png")) +
        list(slides_path.glob("slide*.PNG")) +
        list(slides_path.glob("slide*.png"))
    )

    def extract_slide_number(path: Path) -> int:
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else 9999

    return sorted(slides, key=extract_slide_number)


def go_previous() -> None:
    """Move to the previous slide if possible."""
    if st.session_state.slide_index > 0:
        st.session_state.slide_index -= 1


def go_next(num_slides: int) -> None:
    """Move to the next slide if possible."""
    if st.session_state.slide_index < num_slides - 1:
        st.session_state.slide_index += 1


def render_slide_viewer(slides_dir: str = "slides", title: str = "Presentation Viewer") -> None:
    """Render a Streamlit slide viewer with previous/next controls."""
    st.title(title)

    slide_paths = get_slide_paths(slides_dir)

    if not slide_paths:
        st.error(f"No slide images found in '{slides_dir}'.")
        return

    num_slides = len(slide_paths)

    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0

    current_index = st.session_state.slide_index
    current_slide = slide_paths[current_index]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.button(
            "⬅ Previous",
            on_click=go_previous,
            width='stretch',
            disabled=(current_index == 0),
        )

    with col2:
        st.button(
            "Next ➡",
            on_click=go_next,
            args=(num_slides,),
            width='stretch',
            disabled=(current_index == num_slides - 1),
        )

    st.markdown(f"### Slide {current_index + 1} of {num_slides}")
    st.image(str(current_slide), width='stretch')