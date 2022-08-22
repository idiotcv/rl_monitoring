import os

print(os.path.join(os.path.dirname(__file__), 'white_img'))

os.path.join(os.path.dirname(__file__), './')
print(f"{os.getcwd()}" + "/white_img")