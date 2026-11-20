from langchain_core.output_parsers import StrOutputParser
from src.config import llm
from src.prompts import (
  intent_prompt,
  contextualize_prompt,
  answer_prompt,
  history_prompt,
  general_prompt
)


# Intent Classification chain
intent_chain = (
  intent_prompt|llm|StrOutputParser()
  )

# rewriting prompt chain
contextualize_chain = (
  contextualize_prompt|llm|StrOutputParser()
)

# Document answer chain
document_answer_chain = (
  answer_prompt|llm|StrOutputParser()
)

#History Answer Chain
history_answer_chain = (
  history_prompt|llm|StrOutputParser()
)

# General Conversation Chain
general_chain = (
  general_prompt|llm|StrOutputParser()
)