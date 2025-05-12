import requests
import json
import regex as re


def get_quraan_uthmani(filename=None):
    """Fetches the Qur'an Uthmani edition JSON and save it if `filename` is defined."""
    url = "https://api.alquran.cloud/v1/quran/quran-uthmani"
    try:
        response = requests.get(url)
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {e}")

    if response.ok:
        try:
            content = response.json().get('data')
            if filename:
                with open(filename, "w") as f:
                    f.write(json.dumps(content))
                return content
        except ValueError:
            raise RuntimeError("Failed to parse JSON response")
    else:
        # Attempt to extract an error message from the response JSON
        try:
            error_msg = response.json().get("message", response.text)
        except ValueError:
            error_msg = response.text
        raise RuntimeError(f"Request failed with status code {response.status_code}: {error_msg}")


def get_raw_quraan(surahs, ayat_split_token: str = " ۝ ",
                 surah_split_token: str = " <|endoftext|> "):
    raw_quraan = ""
    for surah in surahs:
        raw_quraan += f"{surah['name']}: "
        for ayah in surah["ayahs"]:
            text = re.sub(r'[\u08D3-\u08E1\u08F0\u0610-\u061A\u06DC-\u06ED]', '', ayah["text"])
            raw_quraan += text.replace('\ufeff', '') + ayat_split_token
        raw_quraan += surah_split_token
    return raw_quraan
