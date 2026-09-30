"""Settings used throughout the project."""

# OWL-ViT prompt sets per amenity. "selected" is the prompt and threshold
# passed on to question generation.
AMENITIES = {
    "bathtub": {
        "prompts": {
            "baseline": ["bathtub"],
            "long": ["white ceramic bathtub in hotel bathroom, rectangular or oval-shaped"],
            "variants": ["bathtub", "white bathtub", "ceramic bathtub", "built-in bathtub",
                         "freestanding bathtub", "hotel bathtub", "bathtub with shower",
                         "shower-over-bathtub", "jacuzzi bathtub"],
        },
        "selected": ("baseline", 0.05),
    },
    "kettle": {
        "prompts": {
            "baseline": ["kettle"],
            "electric": ["electric kettle"],
            "long": ["stainless steel electric kettle"],
            "variants": ["kettle", "electric kettle", "stainless steel kettle", "gooseneck kettle"],
        },
        "selected": ("electric", 0.05),
    },
    "hairdryer": {
        "prompts": {
            "baseline": ["hairdryer"],
            "long": ["black and silver electric hairdryer on a bathroom wall"],
            "variants": ["hairdryer", "electric hair dryer", "wall-mounted hotel hairdryer",
                         "portable travel hair dryer"],
        },
        "selected": ("baseline", 0.05),
    },
    "mirror": {
        "prompts": {
            "baseline": ["mirror"],
            "long": ["Wall-mounted mirror in hotel room or bathroom, framed or round."],
            "variants": ["wall mirror", "bathroom mirror", "bedroom mirror", "hotel mirror",
                         "round mirror", "rectangular mirror", "framed mirror", "vanity mirror",
                         "mirror above sink", "mirror above bed", "mirror on closet door"],
        },
        "selected": ("baseline", 0.15),
    },
    "tv": {
        "prompts": {
            "baseline": ["tv"],
            "long": ["Wall-mounted flat-screen TV in a hotel."],
            "variants": ["tv", "hotel tv", "wall-mounted tv", "flat screen tv", "smart tv",
                         "tv on cabinet", "tv above desk", "tv across bed", "television",
                         "bedroom tv", "living room tv"],
        },
        "selected": ("baseline", 0.05),
    },
}

OWLVIT_MODEL = "google/owlvit-base-patch32"
CONF_SWEEP = [0.05, 0.15, 0.25, 0.35, 0.45]
IOU_THRESHOLD = 0.50
PAD_FRAC = 0.35  # context kept around each detected box

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"]
FINAL_MODEL = "gemini-3-pro-preview"
GENERATION_CONFIG = {"temperature": 0.7, "top_p": 0.95, "max_output_tokens": 2048}

# Traveller profiles, matching the guest / trip-type filters on the search page
PROFILES = {
    "single": "solo traveler",
    "couple": "couple traveling together",
    "group": "large group or family",
}
QUESTIONS_GENERAL = 5
QUESTIONS_PER_PROFILE = 3
