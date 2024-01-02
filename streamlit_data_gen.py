import streamlit as st
# st.title("Data Generation")
left_column, right_column = st.beta_columns(2)
with right_column:
    st.text_input("Temperature", key="temp")

with left_column:
    st.json({
        'fruit': 'apple',
        'book': 'maths',
        'game': 'football'
    })

    st.button("Press")
print(st.session_state.temp)
