from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")


template = ''' Você é um Potter Head!

Aqui está o seu livro preferido {content}

Me tire essa dúida {question}

'''

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------------")
    question = input("Faça a sua pergunta (s para sair): ")
    print("\n\n")

    content = retriever.invoke(question)
    
    result = chain.invoke({"content":content, "question": question})
    print(result)

    if question == "q":
        break


