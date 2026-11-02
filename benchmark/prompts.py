"""Frozen ECG-Agent-Play judge prompts: src/eval at commit f3071e3f."""


def tool_prompt(ecg_id, response_data, model_name, gt_response, category):
    return f"""
You are a medical expert evaluating an AI's ECG dialogue response. Compare the assistant's response to the ground truth.

**Context:**
- ECG ID: {ecg_id}
- Model: {model_name}
- Response Category: {category}
- Tool Output (provided to assistant): {response_data.get('tool_output', 'N/A')}
- User's Question: {response_data.get('context', {}).get('user_query', '')}

**Ground Truth Response:**
{gt_response}

**Assistant Response to Evaluate:**
{response_data.get('response', '')}

**Evaluation Criteria:**
1. Accuracy (1-5): How well does the response match the ground truth? Key information to look for are representative diagnosis classes (e.g., sinus rhythm, myocardial infarction) or measurement interval (e.g., Heart rate, PR interval, QRS duration, QTc interval) for post tool calls and representative key information for direct responses.\x20
    - 5: Fully accurate. If the diagnosis classes mentioned by the model's response is fully accurate, as in all representative diagnosis classes are identified and correct, this score should be applied. For measurement interval, if for example the heart rate, pr interval, qrs duration, and qtc interval is accurate within the normal range for a normal ECG, this score should be applied.\x20
    - 3: Partially accurate. If the diagnosis classes mentioned by the model's response is partially inaccurate, as in one representative diagnosis class is identified but another one is inaccurate (50% accuracy), this score should be applied. For measurement interval, if for example the measurements of heart rate and pr interval is accurate within the normal range for a normal ECG but the QRS duration and QTC interval are inaccurate, this score should be applied.\x20\x20
    - 1: Largely inaccurate. If the diagnosis classes mentioned by the model's response is entirely inaccurate, as in not a single relevant class is identified, this score should be applied. For measurement interval, if for example abnormal heart rate, pr interval, qrs duration, and qtc interval is identified for a completely normal ECG, this score should be applied.
2. Completeness (1-5): Does the response cover essential information mentioned in the ground truth? This score focuses on whether the answer is comprehensive and includes as much essential information as possible.\x20
    - 5: Comprehensive. If the response includes all key information from the ground truth, this score should be applied.\x20
    - 3: Partially complete. If the response includes some key information but is missing others, this score should be applied.
    - 1: Incomplete. If the response is missing all key information, this score should be applied.

**Instructions:** Rate each criterion and provide a brief justification.

**Response Format:**
Accuracy: [score]
Accuracy Explanation: [justification]
Completeness: [score]
Completeness Explanation: [justification]
"""


def direct_prompt(ecg_id, response_data, model_name, gt_response):
    return f"""
You are a medical expert evaluating an AI's ECG dialogue response. Compare the assistant's response to the ground truth.

**Context:**
- ECG ID: {ecg_id}
- Model: {model_name}
- Response Category: direct_response
- User's Question/Previous Turn: {response_data.get('context', {}).get('user_query', '')}

**Ground Truth Response:**
{gt_response}

**Assistant Response to Evaluate:**
{response_data.get('response', '')}

**Evaluation Criteria:**
1. Accuracy (1-5): How well does the response match the ground truth? Key information to look for are representative diagnosis classes (e.g., sinus rhythm, myocardial infarction) or measurement interval (e.g., Heart rate, PR interval, QRS duration, QTc interval) for post tool calls and representative key information for direct responses.\x20
    - 5: Fully accurate. If the diagnosis classes mentioned by the model's response is fully accurate, as in all representative diagnosis classes are identified and correct, this score should be applied. For measurement interval, if for example the heart rate, pr interval, qrs duration, and qtc interval is accurate within the normal range for a normal ECG, this score should be applied.\x20\x20
    - 3: Partially accurate. If the diagnosis classes mentioned by the model's response is partially inaccurate, as in one representative diagnosis class is identified but another one is inaccurate (50% accuracy), this score should be applied. For measurement interval, if for example the measurements of heart rate and pr interval is accurate within the normal range for a normal ECG but the QRS duration and QTC interval are inaccurate, this score should be applied.\x20\x20
    - 1: Largely inaccurate. If the diagnosis classes mentioned by the model's response is entirely inaccurate, as in not a single relevant class is identified, this score should be applied. For measurement interval, if for example abnormal heart rate, pr interval, qrs duration, and qtc interval is identified for a completely normal ECG, this score should be applied.
2. Completeness (1-5): Does the response cover essential information mentioned in the ground truth? This score focuses on whether the answer is comprehensive and includes as much essential information as possible.\x20
    - 5: Comprehensive. If the response includes all key information from the ground truth, this score should be applied.\x20
    - 3: Partially complete. If the response includes some key information but is missing others, this score should be applied.
    - 1: Incomplete. If the response is missing all key information, this score should be applied.
\x20\x20\x20\x20

**Instructions:** Rate each criterion and provide a brief justification.

**Response Format:**
Accuracy: [score]
Accuracy Explanation: [justification]
Completeness: [score]
Completeness Explanation: [justification]
"""


def quality_prompt(dialogue_text, cefr_level):
    return f"""
**Role:** You are an expert evaluator of conversational AI, specializing in linguistics and user experience.

**Task:** Evaluate the provided AI-generated dialogue on two criteria: **Dialogue Naturalness** and **CEFR Level Adherence**.

---

**Dialogue to Evaluate:**
{dialogue_text}


---

**Context for Evaluation:**
- The dialogue is between a user and a medical AI assistant discussing an ECG.
- **The user's language proficiency is CEFR Level: {cefr_level}**

---

**Evaluation Criteria & Scales:**

**1. Dialogue Naturalness (Score 1-5):**
Assess how fluid, coherent, and human-like the conversation feels.
- **5 (Very Natural):** Indistinguishable from a conversation between two humans. Flows logically and has a natural tone.
- **4 (Natural):** Mostly fluid and easy to follow, with only minor instances of stiffness.
- **3 (Acceptable):** Understandable, but contains some awkward phrasing or unnatural transitions that reveal its AI origin.
- **2 (Unnatural):** Stilted, difficult to follow, and clearly sounds robotic or scripted.
- **1 (Very Unnatural):** Incoherent, nonsensical, or fails to resemble a real conversation.

**2. CEFR Adherence (Score 1-5):**
Assess if the assistant's language (vocabulary, sentence structure, jargon) is appropriate for the user's specified CEFR level ({cefr_level}).
- **5 (Excellent Adherence):** Language is perfectly tailored to the user's level. Complex medical concepts are explained simply and appropriately.
- **4 (Good Adherence):** Language is mostly appropriate, with only a few words or phrases being slightly too simple or complex.
- **3 (Moderate Adherence):** A noticeable mismatch. The assistant frequently uses overly complex jargon or, conversely, is condescendingly simple.
- **2 (Poor Adherence):** Consistently mismatched language that would likely hinder the user's understanding or feel patronizing.
- **1 (No Adherence):** The assistant completely ignores the user's language level.

---

CEFR Levels Reference:\x20
- Level A: Describes a user with basic language skills who communicates using short, simple sentences, common vocabulary, and direct questions to express immediate worry and seek simple clarification.
- Level B: Characterizes an intermediate user who employs moderately complex sentences and some medical vocabulary to ask explanatory follow-up questions about the causes, seriousness, and implications of their condition.
- Level C: Defines an advanced user who confidently uses complex sentence structures and precise medical terminology to ask sophisticated, analytical questions about clinical implications, correlations, and prognosis.

**Instructions:** Provide a score and a brief justification for each criterion based on the scales above.

**Required Output Format:**
Naturalness Score: [score]
Naturalness Justification: [one-sentence justification]
CEFR Adherence Score: [score]
CEFR Adherence Justification: [one-sentence justification]
"""


def faithfulness_prompt(tool_output, response_content):
    return f"""
    As an evaluator, your task is to determine if the assistant's response is a faithful representation of the provided tool output.
    You must score the response as either 1 (aligned) or 0 (not aligned).

    **Key Principle for Evaluation:**
    The primary goal is to ensure the response does not misrepresent the tool's output. **It is acceptable for the response to omit some information from the tool_output, as long as the information that *is* presented is accurate and not misleading.** A score of 0 should be given if the response contains incorrect or fabricated details.

    **Evaluation Criteria:**
    - **For classification tools:** The classes mentioned in the response **must be present in the tool_output. The response must not wrongfully represent or invent classes.** It is acceptable if the response does not mention all classes from the tool_output.
    - **For measurement tools:** Any measurements and values mentioned in the response **must be correctly and accurately cited from the tool_output.** It is acceptable to only use a subset of the measurements.
    - **For explanation tools:** The portion of the explanation conveyed in the response **must accurately reflect the core points of the tool_output without contradiction.** The response can be a partial summary and does not need to cover every detail.

    **Tool Output:**
    ```json
    {tool_output}
    ```

    **Assistant Response:**
    ```
    {response_content}
    ```

    **Instructions:**
    Based on the criteria above, is the "Assistant Response" aligned with the "Tool Output"?
    Respond with only a single integer: 1 for aligned, or 0 for not aligned.
    """
