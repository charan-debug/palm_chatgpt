import pandas as pd
import streamlit.components.v1 as components
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from langchain.llms.openai import OpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import time
import os
from mitosheet.streamlit.v1 import spreadsheet
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
from pandasai import SmartDataframe
from pandasai.llm import GooglePalm
import io

# Global variable to store uploaded file
uploaded_file = None

def main():
    global uploaded_file
    st.sidebar.title("App Options")
    option = st.sidebar.selectbox("Choose an option", ["View Instructions", "View Data","Data Profiling","Tableau AI", "CSV Chatbot", "Smart Dataframe Chat"])

    if option == "View Instructions":
        show_instructions()
    elif option == "Data Profiling":
        data_profiling()
    elif option == "CSV Chatbot":
        csv_chatbot()
    elif option == "View Data":
        view_data()
    elif option == "Tableau AI":
        tableau_ai()
    elif option == "Smart Dataframe Chat":
        smart_dataframe_chat()

def show_instructions():
    st.title("Welcome to the AI TOOL - Made for MDH")
    st.write("This tool offers several functionalities to help you analyze and work with your data.")
    st.write("Please select an option from the sidebar to proceed:")
    st.write("- **View Data:** Upload a CSV file and view its contents.")
    st.write("- **Data Profiling:** Upload a CSV file to generate a data profiling report.")
    st.write("- **CSV Chatbot:** Interact with a chatbot to get insights from your CSV data.")
    st.write("- **Tableau AI:** Upload a CSV file to visualize it using Tableau AI.")
    st.write("- **Smart Dataframe Chat:** Chat with a Smart Dataframe powered by Google Palm.")
    st.write("- **View Instructions:** View these instructions again.")

def data_profiling():
    global uploaded_file
    st.title("Data Profiling App")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
            st.stop()  # Stop execution if no file uploaded

    if uploaded_file.name.endswith('.xlsx'):
        # Load Excel file into pandas DataFrame
        df_excel = pd.read_excel(uploaded_file)
        # Save DataFrame as CSV
        csv_filename = uploaded_file.name.replace('.xlsx', '.csv')
        df_excel.to_csv(csv_filename, index=False)
        st.success(f"Excel file converted to CSV: {csv_filename}")
        # Set uploaded file to the converted CSV file
        uploaded_file = open(csv_filename, 'rb')

    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Generate and display the data profile report
    pr = ProfileReport(df, title="Report")
    st_profile_report(pr)

def csv_chatbot():
    global uploaded_file
    st.sidebar.title("OpenAI Settings")
    st.title("Personal Assistant")
    st.text("A BR CREATION")
    st.image("chatbot.jpg", caption="Chatbot", width=178)
   
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
            st.stop()  # Stop execution if no file uploaded

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("You should have an OpenAI API key to continue. Get one at [OpenAI API Keys](https://platform.openai.com/api-keys)")
        st.stop()

    os.environ['OPENAI_API_KEY'] = openai_api_key
    llm = OpenAI(temperature=0)
    agent = create_csv_agent(
        llm,
        uploaded_file,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    predefined_questions = ["How many rows are there in the dataset?", "Explain the dataset."]
    selected_question = st.selectbox("Select a question", ["Select a question"] + predefined_questions) 
    custom_question = st.text_input("Or ask a custom question")

    if st.button("Ask"):
        if selected_question != "Select a question":
            query = selected_question
        elif custom_question.strip() != "":
            query = custom_question.strip()
        else:
            st.warning("Please select a predefined question or ask a custom question.")
            return

        start = time.time()
        answer = agent.run(query)
        end = time.time()
        st.write(answer)
        st.write(f"Answer (took {round(end - start, 2)} s.)")

def view_data():
    global uploaded_file
    st.title("Data Viewer Portal")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
            st.stop()  # Stop execution if no file uploaded

    if uploaded_file.name.endswith('.xlsx'):
        # Load Excel file into pandas DataFrame
        df_excel = pd.read_excel(uploaded_file)
        # Save DataFrame as CSV
        csv_filename = uploaded_file.name.replace('.xlsx', '.csv')
        df_excel.to_csv(csv_filename, index=False)
        st.success(f"Excel file converted to CSV: {csv_filename}")
        # Set uploaded file to the converted CSV file
        uploaded_file = open(csv_filename, 'rb')

    df = pd.read_csv(uploaded_file)

    # Convert the dataframe to a list of dictionaries
    dataframe = df.to_dict(orient="records")

    # Display the dataframe in a Mito spreadsheet
    final_dfs, code = spreadsheet(dataframe)

def tableau_ai():
    global uploaded_file
    st.title("Virtual Tableau AI Tool")
    init_streamlit_comm()

    # Function to get PygWalker HTML
    @st.cache_data
    def get_pyg_html(df: pd.DataFrame) -> str:
        html = get_streamlit_html(df, use_kernel_calc=True, debug=False)
        return html

    # Function to get user uploaded DataFrame
    def get_user_uploaded_data():
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        return None

    if uploaded_file is None:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
            st.stop()  # Stop execution if no file uploaded

    if uploaded_file.name.endswith('.xlsx'):
        # Load Excel file into pandas DataFrame
        df_excel = pd.read_excel(uploaded_file)
        # Save DataFrame as CSV
        csv_filename = uploaded_file.name.replace('.xlsx', '.csv')
        df_excel.to_csv(csv_filename, index=False)
        st.success(f"Excel file converted to CSV: {csv_filename}")
        # Set uploaded file to the converted CSV file
        uploaded_file = open(csv_filename, 'rb')

    df = get_user_uploaded_data()

    if df is not None:
        components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)
    else:
        st.write("Please upload a CSV file to proceed.")

def smart_dataframe_chat():
    st.title("Smart Dataframe Chat")
    
    # Create input field for Google API key
    google_api_key = st.text_input("Enter your Google API key:")
    
    # Check if Google API key is provided
    if google_api_key:
        # Load the Google Palm model with the provided API key
        llm = GooglePalm(api_key=google_api_key)
        
        # Define function to load SmartDataframe
        def load_smart_dataframe(data_file):
            # Load the SmartDataframe
            df = SmartDataframe(data_file, config={"llm": llm})
            return df
        
        # Create file uploader component
        uploaded_file = st.file_uploader("Upload file", type=['csv', 'xlsx'])
        
        # Check if file is uploaded
        if uploaded_file is not None:
            # Check file type
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': # XLSX file
                # Read XLSX file as DataFrame
                xls = pd.ExcelFile(uploaded_file)
                df = pd.read_excel(xls)
                # Convert DataFrame to CSV format
                csv_file = io.StringIO()
                df.to_csv(csv_file, index=False)
                # Load CSV data into a DataFrame
                csv_file.seek(0)
                df = pd.read_csv(csv_file)
            else: # CSV file
                # Load data into a DataFrame
                df = pd.read_csv(uploaded_file)
            
            # Initialize SmartDataframe with the uploaded data
            smart_df = load_smart_dataframe(df)
            
            # Create an input text box for user query
            user_query = st.text_input("Enter your question:")
            
            # Check if the user has entered a query
            if user_query:
                # Use the SmartDataframe to get the response
                response = smart_df.chat(user_query)
                
                # Display the response
                st.write("Response:")
                st.write(response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
