import os

print(os.path.join(os.path.dirname(__file__), 'img'))

os.path.join(os.path.dirname(__file__), './')
print(f"{os.getcwd()}" + "/img")