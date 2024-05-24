import streamlit as st
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class DefTaskExample:
    word: str
    definition: str
    examples: List[str]

@st.cache_resource
def initialize_values(file=None):
    if file is not None:
        with open(file, 'r') as fp:
            return json.load(fp)
    else:
        return {}


def save():
    with open("definition_task_examples.json", 'w') as fp:
        json.dump(def_dict, fp)

def_dict = initialize_values(file=None)

if "current_example" not in st.session_state:

    st.session_state.current_example = DefTaskExample(word='', definition='', examples=[])

curr_word = st.selectbox("Current Slang Word", list(def_dict.keys()))
