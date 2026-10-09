"""The languages measured: every Indian language in FLORES-200, with English as the baseline.

`scheduled` marks the 22 languages of the Eighth Schedule of the Constitution of India.
FLORES-200 has 19 of them; Bodo, Dogri and Konkani are not in it (IN22 has them, but needs a
Hugging Face login). Awadhi, Bhojpuri, Chhattisgarhi and Magahi are in FLORES-200 but not
scheduled; the census counts their speakers under Hindi.

`speakers` is the number of speakers in the Census of India 2011 (scheduled languages, in
millions). Kashmiri is written in two scripts in FLORES; its speakers are counted once, on the
Perso-Arabic entry.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str        # FLORES-200 code
    name: str
    native: str
    script: str
    scheduled: bool
    speakers: float  # millions, Census 2011; 0 where not counted separately


ENGLISH = Language("eng_Latn", "English", "English", "Latin", False, 0.0)

INDIC = [
    Language("hin_Deva", "Hindi", "हिन्दी", "Devanagari", True, 528.35),
    Language("ben_Beng", "Bengali", "বাংলা", "Bengali", True, 97.24),
    Language("mar_Deva", "Marathi", "मराठी", "Devanagari", True, 83.03),
    Language("tel_Telu", "Telugu", "తెలుగు", "Telugu", True, 81.13),
    Language("tam_Taml", "Tamil", "தமிழ்", "Tamil", True, 69.03),
    Language("guj_Gujr", "Gujarati", "ગુજરાતી", "Gujarati", True, 55.49),
    Language("urd_Arab", "Urdu", "اردو", "Perso-Arabic", True, 50.77),
    Language("kan_Knda", "Kannada", "ಕನ್ನಡ", "Kannada", True, 43.71),
    Language("ory_Orya", "Odia", "ଓଡ଼ିଆ", "Odia", True, 37.52),
    Language("mal_Mlym", "Malayalam", "മലയാളം", "Malayalam", True, 34.84),
    Language("pan_Guru", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", True, 33.12),
    Language("asm_Beng", "Assamese", "অসমীয়া", "Bengali", True, 15.31),
    Language("mai_Deva", "Maithili", "मैथिली", "Devanagari", True, 13.58),
    Language("sat_Olck", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "Ol Chiki", True, 7.37),
    Language("kas_Arab", "Kashmiri", "کٲشُر", "Perso-Arabic", True, 6.80),
    Language("kas_Deva", "Kashmiri (Devanagari)", "कॉशुर", "Devanagari", True, 0.0),
    Language("npi_Deva", "Nepali", "नेपाली", "Devanagari", True, 2.93),
    Language("snd_Arab", "Sindhi", "سنڌي", "Perso-Arabic", True, 2.77),
    Language("mni_Beng", "Manipuri", "মৈতৈলোন্", "Bengali", True, 1.76),
    Language("san_Deva", "Sanskrit", "संस्कृतम्", "Devanagari", True, 0.02),
    Language("bho_Deva", "Bhojpuri", "भोजपुरी", "Devanagari", False, 0.0),
    Language("awa_Deva", "Awadhi", "अवधी", "Devanagari", False, 0.0),
    Language("mag_Deva", "Magahi", "मगही", "Devanagari", False, 0.0),
    Language("hne_Deva", "Chhattisgarhi", "छत्तीसगढ़ी", "Devanagari", False, 0.0),
]

ALL = [ENGLISH] + INDIC
BY_CODE = {l.code: l for l in ALL}
