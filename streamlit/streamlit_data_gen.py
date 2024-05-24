import streamlit as st
import openai
import json
# st.title("Data Generation")
# left_column, right_column = st.columns(2)
prompt = "Give me a unique, descriptive sentence using the word \"{}\""


tmp = {
    "a": {"b":[1,2,3,4,5],
          "c":2}
}
def form_callback():
    word = st.session_state.word
    temp = float(st.session_state.temp)
    p = prompt.format(word)
    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": p}], n=1, temperature=temp, max_tokens=50)
    st.session_state.sentence = completion.choices[0]['message']['content']
    st.write(completion.choices[0]['message']['content'])

def accept_callback():
    tmp[st.session_state.question][st.session_state.answer].append(st.session_state.sentence)

# with left_column:
#     # st.text_input("Temperature", key="temp")
#     q = st.selectbox("Question", list(tmp.keys()))
#     answer_choice = st.selectbox("Answer Choice", list(tmp[q].keys()))
#     st.session_state.question = q
#     st.session_state.answer = answer_choice
#     st.write(tmp[q][answer_choice])


# with right_column:
with st.form(key='openai_form'):
    st.text_input("Word", key="word")
    st.text_input("Temperature", key="temp")
    submit_button = st.form_submit_button(label='Submit', on_click=form_callback)

st.button("Accept Sentence", on_click=accept_callback)




    # st.button("Press")
# print(st.session_state.temp)
