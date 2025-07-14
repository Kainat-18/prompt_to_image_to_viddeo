import requests
from together import Together

# Initialize Together client with your API key
client = Together(api_key="tgp_v1_DBS2c9ziPM994zrfkeG2Siu0YJa6TbfDq3NZW8mop94")

# Define the image prompt
prompt = (
    "a chubby girl in the gym with a cat behind workout having bold golden hair, crown in hair, wearing black outfit"
)

# Generate the image
response = client.images.generate(
    prompt=prompt,
    model="black-forest-labs/FLUX.1-dev"
)

# Print full response data
print("Full Response Data:")
print(response.data)

# Try to extract and download the image
try:
    # The response is typically a list of objects
    image_choice = response.data[0]
    image_url = image_choice.url  # Make sure `.url` is accessible

    if image_url:
        print(f"Downloading from: {image_url}")
        image_data = requests.get(image_url).content
        with open("viking_jb_image.jpg", "wb") as f:
            f.write(image_data)
        print("Image saved as 'viking_jb_image.jpg")
    else:
        print("URL was not found in the response object.")
except Exception as e:
    print("Error accessing image URL:", e)

