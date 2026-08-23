import tiktoken

# encoding

text = "Who am i?"

tokenizer = tiktoken.get_encoding("cl100k_base")

tokens = tokenizer.encode(text)

print(tokens)

# decoding

resultIds = [13347, 11, 602, 1097, 2694, 275]  # "Hi, i am sumit"

tokenizer_to_text = tokenizer.decode(resultIds)

print(tokenizer_to_text)
