import streamlit as st
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class TwitterSlangExample:
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

twitter_dict = initialize_values(None)

def save():
    with open("twitter_examples_2023_mengye.json", 'w') as fp:
        json.dump(twitter_dict, fp)

def re_initialize():
    # st.session_state.definition = ''
    st.session_state.examples = []

if "current_example" not in st.session_state:

    st.session_state.current_example = TwitterSlangExample(word='', definition='', examples=[])

if "definition" not in st.session_state:
    st.session_state.definition = ''
if "examples" not in st.session_state:
    st.session_state.examples = []


# left_column, right_column = st.columns(2)

# with left_column:


# with right_column:
curr_word = st.selectbox("Current Slang Word", ["New Word"] + list(twitter_dict.keys()))
if curr_word == "New Word":
    st.text_input("Type Word", key="word")
    st.session_state.current_example = TwitterSlangExample(word=st.session_state.word,
                                                           definition=st.session_state.definition,
                                                           examples=st.session_state.examples)


else:
    curr_ex = twitter_dict[curr_word]
    st.session_state.current_example = TwitterSlangExample(word=curr_word,
                                                           definition=curr_ex['definition'],
                                                           examples=curr_ex['examples'])

# st.json(asdict(st.session_state.current_example))

# if st.session_state.current_example.definition == '':


with st.form(key="examples_form"):
    st.text_input("Input definition for {}".format(st.session_state.current_example.word), key="definition")
    submit_definition = st.form_submit_button(label='Submit Definition')
    if submit_definition:
        st.session_state.current_example = TwitterSlangExample(word=st.session_state.current_example.word,
                                                           definition=st.session_state.definition,
                                                           examples=st.session_state.current_example.examples)
    else:
        st.session_state.current_example = TwitterSlangExample(word=st.session_state.current_example.word,
                                                               definition=st.session_state.current_example.definition,
                                                               examples=st.session_state.current_example.examples)

    st.text_input("Example", key='example')
    submit_example = st.form_submit_button(label='Submit Example')

    if submit_example:
        curr_examples = st.session_state.current_example.examples
        curr_examples.append(st.session_state.example)
        st.session_state.current_example = TwitterSlangExample(word=st.session_state.current_example.word,
                                                               definition=st.session_state.current_example.definition,
                                                               examples=curr_examples)

st.json(asdict(st.session_state.current_example))
if st.button("Save Example"):
    twitter_dict[st.session_state.current_example.word] = {'definition': st.session_state.current_example.definition,
                                                           'examples': st.session_state.current_example.examples}
    print(twitter_dict)
    print("Total number of words = {}".format(len(twitter_dict)))
    re_initialize()


if st.button("Save Data"):

    save()

