"""Prompt templates for Gemini question generation."""

# Amenity-specific examples shown to the model. The "bad" ones steer it away
# from questions the photo already answers.
GENERAL_EXAMPLES = {'bathtub': {'good': ['Is there both a bathtub and a separate shower?',
                      'Does the bathtub appear to be full-size or compact?',
                      'Are the bathroom fixtures modern or outdated?',
                      'Is there enough counter space for toiletries?',
                      'Does the tub have jets or whirlpool features?'],
             'bad': ['Does the bathroom have a sink? (obvious from photo)',
                     'Is this a hotel bathroom? (already known)',
                     'Does it include a toilet? (every bathroom has one)']},
 'tv': {'good': ['Is the TV a smart TV with streaming apps like Netflix or YouTube?',
                 'What is the approximate screen size of the TV?',
                 'Is the TV wall-mounted for optimal viewing, or on a stand?',
                 'Does the TV have cable channels, streaming capabilities, or both?',
                 'Is the TV positioned for comfortable viewing from the bed?'],
        'bad': ['Does the room have a TV? (obvious from photo)',
                'Is the TV electronic? (all TVs are)',
                "Does it have a screen? (that's what TVs are)"]},
 'kettle': {'good': ['Is this an electric kettle or a stovetop model?',
                     'What is the capacity of the kettle - can it boil enough for two cups?',
                     'Does the kettle have temperature control settings or just on/off?',
                     'Is this a rapid-boil kettle or a standard model?',
                     'Are complimentary tea, coffee, or other beverages provided with the kettle?'],
            'bad': ["Does the kettle boil water? (that's its only function)",
                    'Is there a kettle in the room? (obvious from photo)',
                    'Does it have a handle? (all kettles have handles)']},
 'hairdryer': {'good': ['Is the hairdryer wall-mounted or is it a portable handheld model?',
                        'Does the hairdryer appear to be professional-grade or a basic model?',
                        'Does the hairdryer have multiple heat and speed settings?',
                        'Is the power cord long enough for comfortable use at the mirror?',
                        'Does the hairdryer have a diffuser attachment or concentrator nozzle?'],
               'bad': ["Does the hairdryer dry hair? (that's its purpose)",
                       'Is there a hairdryer provided? (visible in photo)',
                       'Does it have a power button? (all hairdryers do)']},
 'mirror': {'good': ['Is this a full-length mirror or just a vanity mirror?',
                     'Does the mirror have built-in lighting or is it well-lit?',
                     'Is there a magnifying mirror for detailed grooming?',
                     'Is the mirror fog-resistant or does it have a defogger?',
                     'Is the mirror positioned at a convenient height and location?'],
            'bad': ['Is there a mirror in the bathroom? (obvious from photo)',
                    "Does the mirror reflect? (that's what mirrors do)",
                    'Is it made of glass? (all mirrors are)']}}

# Examples for the profile-conditioned run (solo / couple / group).
PROFILE_EXAMPLES = {'bathtub': {'single': ['Is the bathtub deep enough for a relaxing solo soak?',
                        'Can I easily reach everything I need while bathing alone?',
                        'Is the tub slip-resistant for safety when showering solo?'],
             'couple': ['Is the bathtub spacious enough for two people?',
                        'Are there dual controls for water temperature?',
                        "Is there enough counter space for both partners' toiletries?"],
             'group': ['Is there both a tub and shower to accommodate multiple people?',
                       'How quickly does the tub fill for multiple consecutive uses?',
                       'Are there safety features like grab bars for children or elderly?']},
 'tv': {'single': ['Can I easily log into my personal streaming accounts?',
                   'Is the TV remote control intuitive for one person?',
                   'Can I adjust the viewing angle from the bed?'],
        'couple': ['Is the screen size adequate for two people viewing from bed?',
                   'Can we both see the screen clearly from different angles?',
                   'Does the TV support multiple user profiles for streaming?'],
        'group': ['Is the TV large enough for multiple people to watch comfortably?',
                  'Can the screen be seen from different seating areas?',
                  'Does it support multiple device connections for group use?']},
 'kettle': {'single': ['Does the kettle have a minimum fill line suitable for one cup?',
                       'Can I easily pour without spilling when making just one drink?',
                       'Does it heat water fast enough for my morning routine?'],
            'couple': ['Can the kettle boil enough water for two drinks at once?',
                       'Are there two mugs or cups provided?',
                       'Is there a variety of tea and coffee options for different tastes?'],
            'group': ['Is the kettle capacity large enough for multiple people?',
                      'How long does it take to boil a full kettle?',
                      'Are there enough mugs for the entire group?']},
 'hairdryer': {'single': ['Is the hairdryer powerful enough for quick solo styling?',
                          'Can I easily style my hair with one hand?',
                          'Does it have the specific settings I need for my hair type?'],
               'couple': ['Is there a second hairdryer or is one sufficient?',
                          'Does it have attachments suitable for different hair types?',
                          'Is the cord long enough for both partners to use comfortably?'],
               'group': ['Are there multiple hairdryers available?',
                         'Is it durable enough for heavy use by multiple people?',
                         'Does it have versatile settings for different hair types?']},
 'mirror': {'single': ['Is the mirror large enough for full outfit checks?',
                       'Can I see myself clearly for detailed makeup or shaving?',
                       'Is there good lighting for solo grooming tasks?'],
            'couple': ['Is the mirror wide enough for two people simultaneously?',
                       'Are there dual sinks with individual mirror space?',
                       'Can both partners see clearly without crowding?'],
            'group': ['Can multiple people use the mirror area at once?',
                      'Is there enough mirror space for a family getting ready?',
                      'Are there additional mirrors in the room for convenience?']}}


def build_general_prompt(label: str, num_questions: int = 5) -> str:
    examples = GENERAL_EXAMPLES.get(label.lower(), GENERAL_EXAMPLES["bathtub"])
    good = "\n".join(f'- "{q}"' for q in examples["good"])
    bad = "\n".join(f'- "{q}"' for q in examples["bad"])
    return f"""You are an AI assistant helping travelers make hotel booking decisions on Trivago.

A guest is viewing this {label} image to evaluate if this hotel accommodation meets their needs.

**Your Task**: Generate {num_questions} practical, decision-relevant questions a traveler would ask about this {label}.

**Requirements**:
1. Focus on FUNCTIONALITY, SIZE, QUALITY, or LUXURY FEATURES
2. Questions should help the guest DECIDE whether to book
3. Use natural language (how people actually talk)
4. Be specific to what's visible in the image
5. Each question should address a different aspect

**Good Examples**:
{good}

**Bad Examples** (avoid these):
{bad}

**Output Format**:
Return ONLY the {num_questions} questions, numbered 1-{num_questions}, one per line. No additional explanation.

Questions:"""


def build_profile_prompt(label: str, num_questions: int = 3) -> str:
    """One call returns questions for all three traveller profiles."""
    return f"""You are an AI assistant helping travelers make hotel booking decisions on Trivago.

Generate {num_questions} questions about this {label} for EACH of these 3 user perspectives:

1. **SOLO TRAVELER**: Questions focused on individual use and convenience
2. **COUPLE**: Questions about suitability for two people traveling together
3. **GROUP/FAMILY**: Questions about capacity and use by multiple people

**Requirements for ALL questions**:
- Focus on FUNCTIONALITY, SIZE, QUALITY, CAPACITY
- Help travelers DECIDE whether to book
- Use natural, conversational language
- Be specific to what's visible in the image
- Each perspective should have DISTINCT concerns

**Output Format** (EXACTLY like this):
SINGLE:
1. [question]
2. [question]
3. [question]

COUPLE:
1. [question]
2. [question]
3. [question]

GROUP:
1. [question]
2. [question]
3. [question]

Generate the questions now:"""
