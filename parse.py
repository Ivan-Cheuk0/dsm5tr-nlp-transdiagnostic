import pdfplumber
import re
import pandas as pd
from tqdm import tqdm

def clean_whitespace(text):
    """Collapses multiple spaces, tabs, and newlines into a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def format_disorder_title(text):
    """Formats disorder titles cleanly, removing overlapping chapter banners without mangling valid names."""
    text = re.sub(r'[–—]', '-', clean_whitespace(text))
    
    # Strip overarching chapter banners only if they precede a distinct disorder title
    chapter_banners = [
        "Neurodevelopmental Disorders",
        "Intellectual Developmental Disorders",
        "Communication Disorders",
        "Motor Disorders",
        "Schizophrenia Spectrum and Other Psychotic Disorders",
        "Bipolar and Related Disorders",
        "Depressive Disorders",
        "Anxiety Disorders",
        "Obsessive-Compulsive and Related Disorders",
        "Trauma- and Stressor-Related Disorders",
        "Dissociative Disorders",
        "Somatic Symptom and Related Disorders",
        "Feeding and Eating Disorders",
        "Elimination Disorders",
        "Sleep-Wake Disorders",
        "Breathing-Related Sleep Disorders",
        "Parasomnias",
        "Sexual Dysfunctions",
        "Disruptive, Impulse-Control, and Conduct Disorders",
        "Substance-Related and Addictive Disorders",
        "Substance-Related Disorders",
        "Alcohol-Related Disorders",
        "Caffeine-Related Disorders",
        "Cannabis-Related Disorders",
        "Hallucinogen-Related Disorders",
        "Inhalant-Related Disorders",
        "Opioid-Related Disorders",
        "Sedative-, Hypnotic-, or Anxiolytic-Related Disorders",
        "Stimulant-Related Disorders",
        "Tobacco-Related Disorders",
        "Other (or Unknown) Substance-Related Disorders",
        "Non-Substance-Related Disorders",
        "Neurocognitive Disorders",
        "Major and Mild Neurocognitive Disorders",
        "Personality Disorders",
        "Cluster A Personality Disorders",
        "Cluster B Personality Disorders",
        "Cluster C Personality Disorders",
        "Other Personality Disorders",
        "Paraphilic Disorders",
        "Other Mental Disorders and Additional Codes",
        "Medication-Induced Movement Disorders and Other Adverse Effects of Medication",
        "Other Conditions That May Be a Focus of Clinical Attention",
        "Conditions for Further Study"
    ]
    
    for banner in chapter_banners:
        if text.lower().startswith(banner.lower() + " ") and len(text) > len(banner) + 5:
            text = text[len(banner):].strip()

    words = text.split(' ')
    formatted = []
    roman_acronyms = {'I', 'II', 'III', 'IV', 'V', 'ADHD', 'PTSD', 'OCD', 'DID', 'REM', 'NREM', 'HIV', 'CO2', 'DSM-5', 'DSM-5-TR'}
    lowercase_words = {'or', 'and', 'of', 'to', 'in', 'with', 'without', 'due', 'by', 'for', 'on', 'at', 'a', 'an', 'the', 'than'}
    
    for idx, w in enumerate(words):
        clean_w = w.upper().strip('(),:;[]')
        if clean_w in roman_acronyms:
            prefix = w[:len(w) - len(w.lstrip('(['))]
            suffix = w[len(w.rstrip('),:;]')):]
            formatted.append(f"{prefix}{clean_w}{suffix}")
        elif idx > 0 and w.lower() in lowercase_words:
            formatted.append(w.lower())
        else:
            formatted.append(re.sub(r"(^|[-/(\[])([a-zA-Z])", lambda m: m.group(1) + m.group(2).upper(), w.lower()))
    return ' '.join(formatted)

def clean_symptom_text(raw_text):
    """Strips PDF artifacts, stray page numbers, and coding notes for BERT vectorisation."""
    text = re.sub(r'\b\d{2,4}\b(?=\s+[A-Z]|\s*$)', '', raw_text)
    text = re.sub(r'\([A-Z]?\d{2,3}\.\d+[A-Z]*\)', '', text)
    text = clean_whitespace(text)
    return text.lower()

def is_valid_title_line(line):
    """Checks if a line prior to 'Diagnostic Criteria' or 'Proposed Criteria' is a genuine disorder title line."""
    if not line or len(line) > 85:
        return False
    if line.endswith('.') or '"' in line or '“' in line or '”' in line or ':' in line:
        return False
    if re.match(r'^([A-Z]\.|[a-z]\.|[a-z]\)|[0-9]+\.|Note|Coding|Specify|If\s|The\s|In\s|For\s|See\s|Common\s|Discussed\s|Independent\s)', line, re.IGNORECASE):
        return False
    blacklist = (
        r'(Functional Consequences|Risk and Prognostic|Differential Diagnosis|'
        r'Comorbidity|Prevalence|Development and Course|Diagnostic Features|'
        r'Diagnostic Markers|Culture-Related|Sex- and Gender-Related|'
        r'Association With Suicidal|Recording Procedures|Specifiers|Subtypes|'
        r'is defined|are characterized|refer to|see the|are not required|'
        r'-Induced Mental Disorders|^Unspecified\s|^Other Specified\s|^Tic Disorders$|'
        r'^SECTION\s+II|^SECTION\s+III|^Emerging Measures)'
    )
    if re.search(blacklist, line, re.IGNORECASE):
        return False
    return True

def reconstruct_disorder_name(history):
    """Walks backward through the line buffer to reconstruct multi-line titles (including Neurocognitive & Section III subtypes)."""
    title_parts = []
    for raw_line in reversed(history):
        # Strip ICD-10/Z/T/R/G codes and blank code lines
        line = re.sub(r'\b[A-Z]\d{2}(\.[A-Z0-9]{1,5})?\b', '', raw_line)
        line = re.sub(r'_{2,}\._{1,}', '', line)
        line = re.sub(r'^\d{1,4}\s+|\s+\d{1,4}$', '', line)
        line = clean_whitespace(line)
        
        if not line:
            continue
            
        if is_valid_title_line(line):
            norm_line = line.lower()
            if title_parts:
                prev_norm = title_parts[0].lower()
                # Stop if the previous line is an identical duplicate or plural section header
                if norm_line == prev_norm or norm_line.rstrip('s') == prev_norm.rstrip('s') or norm_line in prev_norm:
                    break
                starts_continuation = prev_norm.startswith((
                    '(', 'and ', 'or ', 'due ', 'with ', 'without ', 'disorder', 
                    'disorders', 'condition', 'conditions', 'disability)', 'by proxy)',
                    'disease', 'bodies', 'infection', 'injury', 'degeneration', 'etiologies',
                    'syndrome', 'hypomania', 'exposure', 'neurocognitive'
                ))
                ends_incomplete = norm_line.endswith((
                    'and', 'or', 'of', 'to', 'in', 'with', 'due', 'related', 'induced', 
                    'symptom', 'developmental', 'motor', 'another', 'neurocognitive', 
                    'disorder', 'frontotemporal', 'vascular', 'traumatic', 'lewy', 
                    'associated', 'prenatal', 'short-duration', 'mild', 'major', '-'
                ))
                if not (starts_continuation or ends_incomplete):
                    break
            title_parts.insert(0, line)
            if len(title_parts) >= 3:
                break
        else:
            if title_parts:
                break
                
    if title_parts:
        return format_disorder_title(' '.join(title_parts))
    return None

def extract_dsm_symptoms(pdf_path, start_page, end_page):
    letter_pattern = re.compile(r"^\s*([A-Z])\.\s+(.*)")
    number_pattern = re.compile(r"^\s*(\d+)\.\s+(.*)")
    lowercase_pattern = re.compile(r"^\s*([a-i])\.\s+(.*)")
    roman_pattern = re.compile(r"^\s*(i{1,3}|iv|v)\.\s+(.*)")
    
    # Matches both Section II ("Diagnostic Criteria") and Section III ("Proposed Criteria")
    criteria_start = re.compile(r"^\s*(Diagnostic Criteria|Proposed Criteria)", re.IGNORECASE)
    
    explicit_subdisorders = re.compile(
        r"^\s*("
        r"Posttraumatic Stress Disorder in Individuals Older Than 6 Years|"
        r"Posttraumatic Stress Disorder in Children 6 Years and Younger|"
        r"Tourette’s Disorder|Tourette's Disorder|"
        r"Persistent \(Chronic\) Motor or Vocal Tic Disorder|"
        r"Provisional Tic Disorder|"
        r"Catatonia Associated With Another Mental Disorder(?:\s*\(Catatonia Specifier\))?|"
        r"Factitious Disorder Imposed on Self|"
        r"Factitious Disorder Imposed on Another(?:\s*\(Previously Factitious.*)?|"
        r"Gender Dysphoria in Children|"
        r"Gender Dysphoria in Adolescents and Adults|"
        r"in Children|in Adolescents and Adults|"
        r"General Personality Disorder"
        r")(?:\s+[A-Z]\d{2}(?:\.[A-Z0-9]{1,5})?)?\s*$",
        re.IGNORECASE
    )
    
    episode_headers = re.compile(
        r"^\s*(Manic Episode|Hypomanic Episode|Major Depressive Episode)\s*$",
        re.IGNORECASE
    )

    # Case-sensitive check for "Diagnostic Features" so wrapped lowercase "diagnostic features for..."
    # in Neurocognitive Disorder With Lewy Bodies (Criterion C) does not prematurely terminate parsing.
    # Also matches parenthetical "Coding note (see coding table...):" and Section III narrative starts.
    criteria_end = re.compile(
        r"^\s*("
        r"Diagnostic Features|Proposed Diagnostic Features|"
        r"(?i:Specify whether|Specify if|Specify current|Specify the|Specify:|Specifiers|Subtypes|"
        r"Coding and Recording|Recording Procedures|Coding note\b|Prevalence|"
        r"Development and Course|Functional Consequences|"
        r"Various research studies have provided|Alcohol is a neurobehavioral teratogen)"
        r")"
    )
    
    internal_subheaders_end = re.compile(
        r"(?<=[\.:])\s+("
        r"Intrusion Symptoms|Negative Mood|Dissociative Symptoms|Avoidance Symptoms|Arousal Symptoms|"
        r"Persistent Avoidance of Stimuli|Negative Alterations in Cognitions|"
        r"Angry/Irritable Mood|Argumentative/Defiant Behavior|Vindictiveness|"
        r"Aggression to People and Animals|Destruction of Property|Deceitfulness or Theft|Serious Violations of Rules|"
        r"For major neurocognitive disorder:?|For mild neurocognitive disorder:?|"
        r"(?:Probable|Possible)\s+(?:Alzheimer’s|Alzheimer's|frontotemporal|vascular)\s+(?:disease|neurocognitive disorder)\s+is\s+diagnosed.*|"
        r"Major or mild neurocognitive disorder probably due to Parkinson.*|"
        r"For probable major or mild neurocognitive disorder with Lewy bodies.*|"
        r"Coding note\s*\(see.*|"
        r"Various research studies have provided.*|"
        r"Alcohol is a neurobehavioral teratogen.*"
        r")\s*$",
        re.IGNORECASE
    )
    
    standalone_subheaders = re.compile(
        r"^\s*("
        r"Intrusion Symptoms|Negative Mood|Dissociative Symptoms|Avoidance Symptoms|Arousal Symptoms|"
        r"Persistent Avoidance of Stimuli|Negative Alterations in Cognitions|"
        r"Angry/Irritable Mood|Argumentative/Defiant Behavior|Vindictiveness|"
        r"Aggression to People and Animals|Destruction of Property|Deceitfulness or Theft|Serious Violations of Rules|"
        r"For major neurocognitive disorder:?|For mild neurocognitive disorder:?|"
        r"Lifetime experience of at least one major depressive episode.*"
        r")\s*$",
        re.IGNORECASE
    )

    unlettered_preamble_start = re.compile(
        r"^\s*("
        r"A pervasive pattern of|A pervasive and excessive need|"
        r"A problematic pattern of caffeine use|Persistent and recurrent use of the Internet"
        r")\b",
        re.IGNORECASE
    )

    extracted_data = []
    
    current_disorder = "Unknown Disorder"
    base_disorder = "Unknown Disorder"
    in_criteria = False
    in_note = False
    current_buffer = ""
    current_id = ""
    current_type = ""
    parent_number = ""
    parent_lower = ""
    
    line_history = []

    def flush_buffer():
        nonlocal current_buffer
        if current_buffer:
            cleaned_buf = internal_subheaders_end.sub('', current_buffer)
            extracted_data.append({
                "disorder": current_disorder,
                "criterion_id": current_id,
                "criterion_level": "main_criterion" if current_type == "letter" else "sub_symptom",
                "raw_text": clean_whitespace(cleaned_buf),
                "cleaned_text": clean_symptom_text(cleaned_buf)
            })
            current_buffer = ""

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        actual_end_page = min(end_page, total_pages)
        
        for i in tqdm(range(start_page - 1, actual_end_page), desc="Parsing DSM-5 Pages (127-1377)", unit="page"):
            text = pdf.pages[i].extract_text(layout=True)
            if not text:
                continue
            
            for line in text.split('\n'):
                clean_line = clean_whitespace(line)
                if not clean_line or re.match(r'^\d{1,4}$', clean_line):
                    continue
                # Skip standalone ICD-10 code lines inside criteria boxes
                if in_criteria and re.match(r'^[A-Z]\d{2}(\.[A-Z0-9]{1,5})?$', clean_line):
                    continue

                # 1. Check for explicit multi-box sub-disorders
                sub_match = explicit_subdisorders.match(clean_line)
                if sub_match:
                    flush_buffer()
                    matched_text = sub_match.group(1)
                    if matched_text.lower().startswith("in "):
                        raw_sub = f"Gender Dysphoria {matched_text}"
                    else:
                        raw_sub = re.sub(r'\(Previously Factitious.*', '', matched_text, flags=re.IGNORECASE)
                    
                    base_disorder = format_disorder_title(raw_sub)
                    current_disorder = base_disorder
                    in_criteria = True
                    in_note = False
                    line_history.clear()
                    continue

                # 2. Check for Bipolar Episode headers
                ep_match = episode_headers.match(clean_line)
                if ep_match:
                    flush_buffer()
                    sub_name = format_disorder_title(ep_match.group(1))
                    if "Bipolar" in base_disorder:
                        current_disorder = f"{base_disorder} - {sub_name}"
                    in_criteria = True
                    in_note = False
                    continue

                # 3. Check for glued Episode headers at the end of a line
                glued_ep = re.search(r'(?<=[\.\d])\s+(Manic Episode|Hypomanic Episode|Major Depressive Episode)\s*$', clean_line)
                if glued_ep and in_criteria:
                    line_part = clean_line[:glued_ep.start()].strip()
                    if line_part and not in_note:
                        current_buffer += " " + line_part
                    flush_buffer()
                    sub_name = format_disorder_title(glued_ep.group(1))
                    if "Bipolar" in base_disorder:
                        current_disorder = f"{base_disorder} - {sub_name}"
                    in_note = False
                    continue

                # 4. Trigger extraction on "Diagnostic Criteria" or "Proposed Criteria"
                if criteria_start.match(clean_line):
                    flush_buffer()
                    in_criteria = True
                    in_note = False
                    parent_number = ""
                    parent_lower = ""
                    reconstructed = reconstruct_disorder_name(line_history)
                    if reconstructed:
                        base_disorder = reconstructed
                        current_disorder = base_disorder
                    line_history.clear()
                    continue
                
                # Maintain sliding window of history
                line_history.append(clean_line)
                if len(line_history) > 6:
                    line_history.pop(0)
                
                # 5. Turn extraction OFF when hitting narrative sections or specifiers
                if in_criteria and criteria_end.match(clean_line):
                    flush_buffer()
                    in_criteria = False
                    in_note = False
                    continue
                
                if in_criteria:
                    if standalone_subheaders.match(clean_line):
                        continue
                    
                    if re.match(r'^Note:', clean_line, re.IGNORECASE):
                        flush_buffer()
                        in_note = True
                        continue

                    letter_match = letter_pattern.match(clean_line)
                    number_match = number_pattern.match(clean_line)
                    lower_match = lowercase_pattern.match(clean_line)
                    roman_match = roman_pattern.match(clean_line)
                    
                    if letter_match:
                        if re.search(r'\b(M\.D\.|Ph\.D\.|M\.P\.H\.|M\.S\.)\b', clean_line):
                            continue
                        flush_buffer()
                        in_note = False
                        if letter_match.group(1) == 'A' and " - " in current_disorder and "Criteria have been met" in clean_line:
                            current_disorder = base_disorder
                        current_id = letter_match.group(1)
                        current_type = "letter"
                        parent_number = ""
                        parent_lower = ""
                        current_buffer = internal_subheaders_end.sub('', clean_line)
                    elif not current_buffer and not parent_number and unlettered_preamble_start.match(clean_line):
                        # Capture un-lettered Criterion A preambles (Cluster B/C Personality Disorders, Caffeine Use, Internet Gaming)
                        flush_buffer()
                        in_note = False
                        current_id = "A"
                        current_type = "letter"
                        parent_number = ""
                        parent_lower = ""
                        current_buffer = "A. " + internal_subheaders_end.sub('', clean_line)
                    elif number_match:
                        flush_buffer()
                        in_note = False
                        parent_number = number_match.group(1)
                        parent_lower = ""
                        current_id = parent_number
                        current_type = "number"
                        current_buffer = internal_subheaders_end.sub('', clean_line)
                    elif roman_match and parent_number and parent_lower:
                        flush_buffer()
                        in_note = False
                        current_id = f"{parent_number}{parent_lower}{roman_match.group(1)}"
                        current_type = "roman"
                        current_buffer = internal_subheaders_end.sub('', clean_line)
                    elif lower_match and (parent_number or in_note):
                        flush_buffer()
                        in_note = False
                        parent_lower = lower_match.group(1)
                        current_id = f"{parent_number}{parent_lower}" if parent_number else parent_lower
                        current_type = "lowercase"
                        current_buffer = internal_subheaders_end.sub('', clean_line)
                    elif current_buffer and not in_note:
                        current_buffer += " " + internal_subheaders_end.sub('', clean_line)

    flush_buffer()
    return extracted_data

if __name__ == "__main__":
    PDF_FILE_PATH = "DSM 5 TR-APA (2022).pdf" 
    START_PAGE = 127 
    END_PAGE = 1377
    
    try:
        print(f"\nStarting extraction pipeline from page {START_PAGE} to {END_PAGE}...")
        dataset = extract_dsm_symptoms(PDF_FILE_PATH, START_PAGE, END_PAGE)
        
        df = pd.DataFrame(dataset)
        df.to_json("dsm_symptoms_cleaned_full.json", orient="records", indent=4)
        print(f"\nExtraction complete. Saved {len(df)} criteria across {df['disorder'].nunique()} disorders to 'dsm_symptoms_cleaned_full.json'.")
        
    except FileNotFoundError:
        print(f"\nError: Could not find '{PDF_FILE_PATH}'. Please check the file path.")
