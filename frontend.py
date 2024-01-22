from typing import Dict, Optional, Union, List

import streamlit as st
from orca3b import main as orca
from phi2 import main as phi2prompt
# from openllama3bv2 import main as open_llama

from memory import ChatMemory as mem

def initialize_memory() -> mem:
    """Initialize ChatMemory."""
    return mem()

m: mem = initialize_memory()

def set_page_configuration():
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="SQL2TEXT",
        page_icon=":lama:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

def retrieve_clicked_button(button_states: Dict[str, bool]) -> Optional[str]:
    """Retrieve the clicked button from button_states."""
    for key, value in button_states.items():
        if value:
            return key
    return None

def get_difficulty_label(value: int) -> str:
    """Get difficulty label based on the difficulty level."""
    difficulty_labels = {
        1: f'1: "Simple, like asking your GPU to render a video", temperature = {-0.1 + (((1 + 1) / 10)):.1f}, top_p = {1.20 - (((1 + 1) / 10)):.1f}, top_k = {(1 * 10):.1f}',
        2: f'2: "Easy, like requesting your GPU to play a video game", temperature = {-0.1 + (((2 + 1) / 10)):.1f}, top_p = {1.20 - (((2 + 1) / 10)):.1f}, top_k = {(2 * 10):.1f}',
        3: f'3: "Intermediate, like convincing your GPU to simulate basic weather", temperature = {-0.1 + (((3 + 1) / 10)):.1f}, top_p = {1.20 - (((3 + 1) / 10)):.1f}, top_k = {(3 * 10):.1f}',
        4: f'4: "Advanced, like coaxing your GPU to render a low-poly landscape", temperature = {-0.1 + (((4 + 1) / 10)):.1f}, top_p = {1.20 - (((4 + 1) / 10)):.1f}, top_k = {(4 * 10):.1f}',
        5: f'5: "Challenging, like persuading your GPU to process complex fluid dynamics", temperature = {-0.1 + (((5 + 1) / 10)):.1f}, top_p = {1.20 - (((5 + 1) / 10)):.1f}, top_k = {(5 * 10):.1f}'
    }
    return difficulty_labels.get(value, "Unknown")

def query():
    """Handle user input and generate responses."""
    # Input text area
    st.write("Enter your question:")
    user_input: str = st.text_area("Question", key="user_input", height=180)
    schema: str = st.text_area("Schema", key="schema", height=420)
    # Difficulty level slider
    difficulty_level: int = st.slider(
        "Creativity Index:",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    # Display the selected difficulty level and its label
    difficulty_label: str = get_difficulty_label(difficulty_level)
    st.write(f"Selected Difficulty Level: {difficulty_level} - {difficulty_label}")

    # Generate response button
    if st.button("Generate Response") and len(user_input) >= 10:
        # Display a loading spinner while generating text
        with st.spinner("Generating Response..."):
            phiresponse: List[str] = phi2prompt(user_input, schema, difficulty_level)
            orcaresp: List[str] = orca(user_input, schema, difficulty_level)
            # llamaresp = open_llama(user_input, schema, difficulty_level)

        # Display responses
        st.write("Phi-2")
        for query in phiresponse:
            st.code(query, language='SQL')
        st.write("orca_mini_3b")
        for query in orcaresp:
            st.code(query, language='SQL')

        m.write_to_file(f"Question:\n{user_input}\n\nAnswer:\nPhi-2:\n{phiresponse}\nOrca:\n{orcaresp}")

def display_chat_history(chats: List[str]):
    """Display chat history or handle new chat."""
    st.sidebar.title('Chat History')
    button_states: Dict[str, bool] = {}

    for chat in chats:
        button_states[chat] = st.sidebar.button(
            f'{chat.strip(".log")}', key=chat)

    clicked_button: Optional[str] = retrieve_clicked_button(button_states)

    if clicked_button:
        content: str = m.read_log_file(clicked_button)
        content: List[str] = content.split('\n')
        content: List[str] = [item for item in content if item != ""]
        st.write(content[0:])
    else:
        query()

def main():
    """Main function to execute the Streamlit app."""
    set_page_configuration()

    st.title("TEXT2SQL Self Healing Small Models Chaining")
    st.markdown(
        """
        <style>
            .stButton>button {
                width: 100%;
                border-radius: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    new_chat: bool = st.sidebar.button(f'New Chat', key='Search')

    chats: List[str] = m.list_log_files()
    display_chat_history(chats)

if __name__ == "__main__":
    main()
