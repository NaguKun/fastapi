import requests
import base64

url = "http://127.0.0.1:7860"

payload = {
   "prompt": "modern luxury villa, coconut trees, greenery landscape, along the white beach, clear sky, day time, warm lighting RAW Photo, RAW texture, Super Realistic, 32K UHD, DSLR, soft lighting, high quality, film rating, Fujifilm XT3 <lora:AARG_villa-000015:0.8> <lora:add_detail:0.3> <lora:epi_noiseoffset2:0.3>",
   "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
    "width": 512,
    "height": 512,
    "seed": 123456789,
 }
response = requests.post(f"{url}/sdapi/v1/txt2img", json=payload)

r = response.json()
print(r)  # See the actual response structure

if 'images' in r:
    with open("testapi/output3.png", "wb") as f:
        f.write(base64.b64decode(r['images'][0]))
    print("✅ Image saved as output3.png")
else:
    print("❌ No 'images' key in response. Full response:")
    print(r)