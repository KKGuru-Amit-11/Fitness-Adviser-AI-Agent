# Import Reqiured Library

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from crewai import Agent,Task,Crew,Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool

# # LLM Monitering
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT']='HealthCare Fitness Adviser AI Agents'


# Creating Web Page Header
st.subheader('**Multi AI Agents HealthCare Fitness Advisor**')

# Getting Task from Web
with st.form(key='Submission_Form',clear_on_submit=True):
    name=st.text_input(label='**Write Your Name:**')
    gender=st.selectbox(label='**Select Your Gender:**',
                        options=['Male','Female'],index=None)
    age=st.slider(label='**Select Your Age:**',min_value=0,max_value=100,value=18,step=1)
    disease=st.selectbox(label='**Have you Disease:**',
                         options=['Yes','No'],index=None)
    llm_model_name=st.selectbox(label='**Select LLM Model:**',
                              options=['Gemini Model','Llama Model'],index=None)
    submit_button=st.form_submit_button('Submit:')
    if submit_button:
        st.info('Input Details:')
        st.markdown(f'Patient Name: {name}...')
        st.markdown(f'Patient Gender: {gender}...')
        st.markdown(f'Patient Age: {age}...')
        st.markdown(f'Patient Disease: {disease}...')
        st.markdown(f'Select Model: {llm_model_name}...')

# Creating LLM Variable
def model_selection(value):
    if value == 'Gemini Model':
        os.environ['GOOGLE_API_KEY']=st.secrets['GOOGLE_API_KEY']
        llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',api_key=os.getenv('GOOGLE_API_KEY'))
        return llm_model
    else:
        os.environ['GROQ_API_KEY']=st.secrets['GROQ_API_KEY']
        llm_model = ChatGroq(model='llama3-8b-8192',api_key=os.getenv('GROQ_API_KEY'))
        return llm_model

LLM_Model=model_selection(llm_model_name)

def disease_func(value):
    if value == "Yes":
        input_value = st.text_input("**Please Specify Your Disease:**")
        button=st.button('Submit:')
        if button:
            st.warning(f'**Patient Disease: {input_value}...**')
        return input_value
    else:
        return disease

disease_name=disease_func(disease)

# Initialize WebSearch Tool
os.environ['SERPER_API_KEY']=st.secrets['SERPER_API_KEY']
search_tool = SerperDevTool()

if disease=="Yes":
    fitness_expert_agent = Agent(
        role='Fitness Expert',
        goal='''{name} is a customer Analyze the his Fitness requirments. {name} is {age} year old. 
        {gender} with disease of {disease_name}. Suggest Exercise routines and fitness Strategies''',
        backstory='''Expert the understanding fitness needs, age-specific requirment 
        and {gender}-Specific consideration.skilled in developing customized Exercise 
        routines and fitness strategies''',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=LLM_Model
    )

    nutritionist_expert_agent = Agent(
        role='Nutritionist Expert',
        goal='''{name} is a customer Assess nutritional his requirements. {name} is {age}-year-old 
        and {gender} with disease of {disease_name} and provide dietary recommendations''',
        backstory='''Knowledgeable in nutrition for different age groups and genders, 
        especially for individuals of {age} years old and {gender}. Provides tailored 
        dietary advice based on specific nutritional needs''',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=LLM_Model
    )

    senior_doctor_agent = Agent(
        role='Senior Doctor',
        goal='''{name} is a patient Evaluate the overall health considerations. 
        {name} is {age}-year-old {gender} with disease of {disease_name} and provide 
        recommendations for a healthy lifestyle. Provide recommendations for managing {disease_name}''',
        backstory='''Medical professional experienced in assessing overall health and 
        well-being. Offers recommendations for a healthy lifestyle considering age, 
        gender, and disease factors''',
        verbose=True,
        allow_delegation=True,
        llm=LLM_Model
    )
    
    disease_expert_agent = Agent(
        role="Disease Expert",
        goal="""Provide recommendations for managing {disease_name}""",
        backstory="""Specialized in dealing with individuals having {disease_name}. 
        Offers tailored advice for managing the specific health condition. 
        Do not prescribe medicines but only give advice.""",
        verbose=True,
        allow_delegation=True,
        llm=LLM_Model
    )

    # Define Tasks with Disease Expert
    fitness_expert_task = Task(
        description="""Analyze the fitness requirements for a {age}-year-old {gender}. 
        Provide recommendations for exercise routines and fitness strategies.""",
        expected_output='''A personalized workout plan, specifying exercises, sets, reps, rest times, and duration. 
        Progress tracking, offering performance feedback and adjustments to routines. 
        Motivational tips to keep users engaged. 
        Recommendations on proper form, warm-up, and cool-down routines. 
        Real-time modifications based on user-reported conditions (e.g., soreness, injury).''',
        agent=fitness_expert_agent
    )

    nutritionist_expert_task = Task(
        description="""Assess nutritional requirements for a {age}-year-old {gender}. 
        Provide dietary recommendations based on specific nutritional needs. 
        Do not prescribe a medicine""",
        expected_output='''A detailed meal plan with recipes, portion sizes, and nutritional breakdown (macros, vitamins, and minerals). 
        Suggestions for supplements or additional nutrients. 
        Adjustments to dietary plans based on user feedback (e.g., allergies, preferences). 
        Healthy eating tips and alternatives to user’s current eating habits. 
        Insights into how nutrition influences energy, recovery, and overall fitness progress.''',
        agent=nutritionist_expert_agent
    )

    senior_doctor_task = Task(
        description="""Evaluate overall health considerations for a {age}-year-old {gender}. 
        Provide recommendations for a healthy lifestyle.""",
        expected_output='''Medical guidance on fitness activities, including safe exercises based on user health conditions. 
        Warnings about potential health risks (e.g., if a user reports symptoms that could indicate injury or overtraining). 
        Recommendations on recovery routines or injury management. 
        Insights on how to balance fitness and health for users with chronic conditions (e.g., arthritis, hypertension). 
        When necessary, advice to consult a doctor or medical professional based on health changes or symptoms''',
        agent=senior_doctor_agent
    )
    disease_task = Task(
        description="""Provide recommendations for managing {disease_name}""",
        expected_output='''Disease-specific fitness and nutrition guidelines (e.g., how diabetes impacts exercise intensity or meal timing). 
        Recommendations for managing energy levels and symptoms related to specific conditions. 
        Alerts for when certain exercises or foods may exacerbate symptoms of a user’s condition. 
        Detailed educational content about how diseases interact with lifestyle choices (e.g., how cardiovascular health influences fitness outcomes). 
        Suggestions for modifications to exercise or nutrition plans based on disease progression or changes in symptoms.''',
        agent=disease_expert_agent
    )

    crew = Crew(
        agents=[fitness_expert_agent, nutritionist_expert_agent, 
                senior_doctor_agent,disease_expert_agent],
        tasks=[fitness_expert_task, nutritionist_expert_task, 
                senior_doctor_task,disease_task],
        verbose=True,
        process=Process.sequential,
        manager_llm=LLM_Model
    )
else:
    fitness_expert_agent = Agent(
        role='Fitness Expert',
        goal='''{name} is a customer Analyze the his Fitness requirments. {name} is {age} 
        year old and {gender}. Suggest Exercise routines and fitness Strategies''',
        backstory='''Expert the understanding fitness needs, age-specific requirment 
        and {gender}-Specific consideration.skilled in developing customized Exercise 
        routines and fitness strategies''',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=LLM_Model
    )

    nutritionist_expert_agent = Agent(
        role='Nutritionist Expert',
        goal='''{name} is a customer Assess nutritional his requirements. {name} is 
        {age}-year-old and {gender}. provide dietary recommendations''',
        backstory='''Knowledgeable in nutrition for different age groups and genders, 
        especially for individuals of {age} years old and {gender}. Provides tailored 
        dietary advice based on specific nutritional needs''',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=LLM_Model
    )

    senior_doctor_agent = Agent(
        role='Senior Doctor',
        goal='''{name} is a patient Evaluate the overall health considerations. 
        {name} is {age}-year-old {gender}. provide recommendations for a healthy lifestyle.''',
        backstory='''Medical professional experienced in assessing overall health and 
        well-being. Offers recommendations for a healthy lifestyle considering age, 
        gender, and disease factors''',
        verbose=True,
        allow_delegation=True,
        llm=LLM_Model
    )

    # Define Tasks with Disease Expert
    fitness_expert_task = Task(
        description="""Analyze the fitness requirements for a {age}-year-old {gender}. 
        Provide recommendations for exercise routines and fitness strategies.""",
        expected_output='''A personalized workout plan, specifying exercises, sets, reps, rest times, and duration. 
        Progress tracking, offering performance feedback and adjustments to routines. 
        Motivational tips to keep users engaged. 
        Recommendations on proper form, warm-up, and cool-down routines. 
        Real-time modifications based on user-reported conditions (e.g., soreness, injury).''',
        agent=fitness_expert_agent
    )

    nutritionist_expert_task = Task(
        description="""Assess nutritional requirements for a {age}-year-old {gender}. 
        Provide dietary recommendations based on specific nutritional needs. 
        Do not prescribe a medicine""",
        expected_output='''A detailed meal plan with recipes, portion sizes, and nutritional breakdown (macros, vitamins, and minerals). 
        Suggestions for supplements or additional nutrients. 
        Adjustments to dietary plans based on user feedback (e.g., allergies, preferences). 
        Healthy eating tips and alternatives to user’s current eating habits. 
        Insights into how nutrition influences energy, recovery, and overall fitness progress.''',
        agent=nutritionist_expert_agent
    )

    senior_doctor_task = Task(
        description="""Evaluate overall health considerations for a {age}-year-old {gender}. 
        Provide recommendations for a healthy lifestyle.""",
        expected_output='''Medical guidance on fitness activities, including safe exercises based on user health conditions. 
        Warnings about potential health risks (e.g., if a user reports symptoms that could indicate injury or overtraining). 
        Recommendations on recovery routines or injury management. 
        Insights on how to balance fitness and health for users with chronic conditions (e.g., arthritis, hypertension). 
        When necessary, advice to consult a doctor or medical professional based on health changes or symptoms''',
        agent=senior_doctor_agent
    )

    crew = Crew(
        agents=[fitness_expert_agent, nutritionist_expert_agent, senior_doctor_agent],
        tasks=[fitness_expert_task, nutritionist_expert_task, senior_doctor_task],
        verbose=True,
        process=Process.sequential,
        manager_llm=LLM_Model
    )


# Input ParaMeter
health_input={
    'name':name,
    'gender':gender,
    'age':age,
    'disease_name':disease_name
}

if st.button('Generate'):
    with st.spinner('Generate Response...'):
        result=crew.kickoff(inputs=health_input)
        res=str(result)
        st.info('Here is Response')
        st.markdown(result)
        st.download_button(label='Download Text File',
                           file_name=f'{name} Fitness Advise.txt',data=res)