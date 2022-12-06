import base64
with open("./source_images/1.jpg","rb") as f:
    data = f.read()
image = base64.b64encode(data)
print(image)