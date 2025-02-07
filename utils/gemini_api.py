import requests

#### Basic Gemini API functions based on Google API REST API so we have seed support 
# without using Vertex AI (to bypass 5 RPM )
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"

class GeminiAPI:
    def __init__(self, api_key, base_url=BASE_URL, **kwargs):
       self.api_key = api_key
       self.base_url = base_url
       self._cached_contents = {}

    def _get_generate_content_url(self, model_name):
        url = self.base_url + "models/" + model_name
        url += f":generateContent?key={self.api_key}"
        return url
    
    def _get_cache_content_url(self):
        return self.base_url + f"cachedContents?key={self.api_key}"
    
    def cache_prompt(self, model_name, prompt, time=300):
        url = self._get_cache_content_url()
        data = {
            "model": f"models/{model_name}",
            "system_instruction": { "parts" : { "text": prompt }},
            "ttl": f"{time}s"
        }
        headers = { 'Content-Type': "application/json"}
        r = requests.request("POST", url=url, headers=headers, json=data)
        r = r.json()
        # here: get cached name
        self._cached_contents[model_name] = r["name"]

    def generate_content(self, model_name, batch, prompt=None):
        headers = { 'Content-Type': "application/json"}
        messages = [
            {"role": "user", "parts": [{"text": "\n".join(batch)}]}
        ]
        data = {
            "generationConfig": { "seed": 0 },
            "contents": messages
        }
        cache_name = self._cached_contents.get(model_name, None)
        if cache_name is None:
            data.update(
                {"system_instruction": { "parts" : { "text": prompt }}}
            )
        else:
            data.update(
                {"cachedContent": f"{cache_name}"}
            )
        
        url = self._get_generate_content_url(model_name)
        r = requests.request("POST", url=url, headers=headers, json=data)
        r = r.json()
        text = r["candidates"][0]["content"]["parts"][0]["text"].strip(" \n")
        return text
    
# Vertex AI
    # vertexai.init(project="gen-lang-client-0652007067", location="europe-west9")
    # vertexai.init(api_key=os.getenv("GOOGLE_GENAI_KEY"))
# Gemini API SDK Python 
    # genai.configure(api_key=os.getenv("GOOGLE_GENAI_KEY"))
    # config = genai.GenerationConfig(seed=0) # not supported yet
    # client = GenerativeModel(model_name, system_instruction=prompt, generation_config={"seed":0})