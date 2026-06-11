import json
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# параметры из arXiv:2606.01657, пересказ своими словами
article = (
    "We characterize a microring resonator on a thin-film lithium tantalate "
    "platform. The ring radius is 60 um and the ring-to-bus coupling gap is "
    "0.45 um. The measured free spectral range is about 350 GHz, and the "
    "loaded quality factor of the pump mode is 1000000. Photon pairs are "
    "generated across the telecom band from 1510 to 1570 nm."
)

system = (
    "You extract device parameters from photonics papers. "
    "Respond ONLY with valid JSON, no explanations, no markdown."
)
user = (
    "Extract these fields:\n"
    "- platform (string: the material platform)\n"
    "- ring_radius_um (number)\n"
    "- coupling_gap_um (number)\n"
    "- fsr_ghz (number)\n"
    "- loaded_q (number, plain float like 1000000)\n"
    "- wavelength_min_nm (number)\n"
    "- wavelength_max_nm (number)\n"
    "Use null if a field is missing.\n\nText:\n" + article
)

resp = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    temperature=0,
)

raw = resp.choices[0].message.content
print("RAW:", raw)

class RingParams(BaseModel):
    platform: str | None
    ring_radius_um: float | None
    coupling_gap_um: float | None
    fsr_ghz: float | None
    loaded_q: float | None
    wavelength_min_nm: float | None
    wavelength_max_nm: float | None

clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
data = json.loads(clean)
result = RingParams.model_validate(data)
print("PARSED:", result)
