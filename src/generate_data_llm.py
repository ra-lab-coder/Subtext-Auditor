"""
LLM-based synthetic data generator for DeepSea Communication Orientation Auditor.

Generates paired conversations (same scenario, two styles) using Google Gemini API.
Each API call produces both task-oriented and emotionally-dependent versions.

Installation:
    pip install google-generativeai

Environment:
    export GEMINI_API_KEY="your-api-key-here"
    Get your key from: https://makersuite.google.com/app/apikey
"""

import os
import csv
import json
import uuid
import random
import time
import argparse
import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# Try to import Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed.")
    print("Install with: pip install google-generativeai")

# Built-in neutral scenarios (fallback if JSON file doesn't exist)
BUILTIN_SCENARIOS = [
    {"scenario_id": "scenario_001", "description": "discussing a work deadline", "setting": "coworkers", "difficulty": "easy"},
    {"scenario_id": "scenario_002", "description": "asking for help with a technical problem", "setting": "coworkers", "difficulty": "easy"},
    {"scenario_id": "scenario_003", "description": "sharing concerns about a job interview", "setting": "coworkers", "difficulty": "medium"},
    {"scenario_id": "scenario_004", "description": "discussing a family situation", "setting": "friends", "difficulty": "medium"},
    {"scenario_id": "scenario_005", "description": "managing school assignments", "setting": "classmates", "difficulty": "easy"},
    {"scenario_id": "scenario_006", "description": "navigating a conflict with a roommate", "setting": "friends", "difficulty": "medium"},
    {"scenario_id": "scenario_007", "description": "seeking advice on career transition", "setting": "coworkers", "difficulty": "hard"},
    {"scenario_id": "scenario_008", "description": "one person had a difficult day and reaches out", "setting": "friends", "difficulty": "hard"},
    {"scenario_id": "scenario_009", "description": "preparing for an important presentation", "setting": "coworkers", "difficulty": "medium"},
    {"scenario_id": "scenario_010", "description": "dealing with work challenges", "setting": "coworkers", "difficulty": "hard"},
    {"scenario_id": "scenario_011", "description": "managing work-life balance", "setting": "coworkers", "difficulty": "medium"},
    {"scenario_id": "scenario_012", "description": "preparing for an upcoming exam", "setting": "classmates", "difficulty": "easy"},
    {"scenario_id": "scenario_013", "description": "discussing a personal conflict", "setting": "friends", "difficulty": "hard"},
    {"scenario_id": "scenario_014", "description": "seeking feedback on a project proposal", "setting": "coworkers", "difficulty": "easy"},
    {"scenario_id": "scenario_015", "description": "one person shares concerns about their situation", "setting": "friends", "difficulty": "hard"},
    {"scenario_id": "scenario_016", "description": "planning a collaborative project", "setting": "coworkers", "difficulty": "easy"},
    {"scenario_id": "scenario_017", "description": "struggling with motivation and productivity", "setting": "classmates", "difficulty": "medium"},
    {"scenario_id": "scenario_018", "description": "one person reaches out after not talking for a while", "setting": "friends", "difficulty": "hard"},
    {"scenario_id": "scenario_019", "description": "reviewing code or technical documentation together", "setting": "coworkers", "difficulty": "easy"},
    {"scenario_id": "scenario_020", "description": "navigating a misunderstanding", "setting": "friends", "difficulty": "hard"},
]


def load_scenarios(scenarios_path: str, min_scenarios: int = 100, auto_generate: bool = True) -> List[Dict]:
    """
    Load scenarios from JSON file, or generate/return built-in list if file doesn't exist.
    
    Args:
        scenarios_path: Path to scenarios JSON file
        min_scenarios: Minimum number of scenarios needed (used for auto-generation)
        auto_generate: If True, automatically generate scenarios if file doesn't exist
    
    Returns:
        List of scenario dictionaries
    """
    if os.path.exists(scenarios_path):
        try:
            with open(scenarios_path, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
            if not isinstance(scenarios, list):
                raise ValueError(f"Scenarios file must contain a JSON array, got {type(scenarios)}")
            print(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")
            return scenarios
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {scenarios_path}: {e}")
            if auto_generate:
                print(f"Auto-generating {min_scenarios} scenarios...")
                return generate_large_scenario_set(scenarios_path, min_scenarios)
            print("Falling back to built-in scenarios")
            return BUILTIN_SCENARIOS
        except Exception as e:
            print(f"Warning: Error loading {scenarios_path}: {e}")
            if auto_generate:
                print(f"Auto-generating {min_scenarios} scenarios...")
                return generate_large_scenario_set(scenarios_path, min_scenarios, seed=42)
            print("Falling back to built-in scenarios")
            return BUILTIN_SCENARIOS
    
    # File doesn't exist
    if auto_generate:
        print(f"Scenarios file not found: {scenarios_path}")
        print(f"Auto-generating {min_scenarios} scenarios...")
        return generate_large_scenario_set(scenarios_path, min_scenarios, seed=42)
    else:
        print(f"Using built-in scenarios (file not found: {scenarios_path})")
        return BUILTIN_SCENARIOS


def generate_large_scenario_set(output_path: str = "data/scenarios.json", n_scenarios: int = 100, seed: int = 42):
    """
    Generate a larger neutral scenario set and save to JSON.
    This is a helper function to create diverse, neutral scenarios.
    
    Args:
        output_path: Where to save the scenarios JSON
        n_scenarios: Number of scenarios to generate
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    settings = ["coworkers", "classmates", "friends"]
    difficulties = ["easy", "medium", "hard"]
    
    # Neutral scenario templates (avoiding emotional dependency cues)
    scenario_templates = [
        "discussing a work deadline",
        "asking for help with a technical problem",
        "sharing concerns about a job interview",
        "discussing a family situation",
        "managing school assignments",
        "navigating a conflict",
        "seeking advice on career transition",
        "one person had a difficult day and reaches out",
        "preparing for an important presentation",
        "dealing with work challenges",
        "managing work-life balance",
        "preparing for an upcoming exam",
        "discussing a personal conflict",
        "seeking feedback on a project proposal",
        "one person shares concerns about their situation",
        "planning a collaborative project",
        "struggling with motivation and productivity",
        "one person reaches out after not talking for a while",
        "reviewing code or technical documentation together",
        "navigating a misunderstanding",
        "discussing a project timeline",
        "asking for input on a decision",
        "sharing updates about a situation",
        "coordinating on a task",
        "discussing a problem that came up",
        "one person contacts the other about something",
        "seeking clarification on an issue",
        "discussing next steps",
        "sharing information about an event",
        "asking for perspective on a matter",
        "working through a technical challenge",
        "brainstorming solutions to a problem",
        "reviewing progress on a shared goal",
        "discussing changes to a plan",
        "seeking advice on a professional decision",
        "coordinating schedules and availability",
        "discussing resource allocation",
        "sharing observations about a situation",
        "working on improving a process",
        "discussing expectations and deliverables",
        "navigating a scheduling conflict",
        "discussing priorities and trade-offs",
        "sharing relevant information",
        "coordinating logistics",
        "discussing potential approaches",
    ]
    
    scenarios = []
    used_combinations = set()  # Track to avoid exact duplicates
    
    for i in range(1, n_scenarios + 1):
        # Try to get unique combinations
        attempts = 0
        while attempts < 100:
            template = random.choice(scenario_templates)
            setting = random.choice(settings)
            difficulty = random.choice(difficulties)
            combo = (template, setting, difficulty)
            
            if combo not in used_combinations or len(used_combinations) >= len(scenario_templates) * len(settings) * len(difficulties):
                used_combinations.add(combo)
                break
            attempts += 1
        
        scenarios.append({
            "scenario_id": f"scenario_{i:03d}",
            "description": template,
            "setting": setting,
            "difficulty": difficulty
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(scenarios)} scenarios and saved to {output_path}")
    return scenarios


def create_prompt(scenario: Dict, samples_per_scenario: int = 2, hard_negative_ratio: float = 0.3) -> str:
    """
    Create the prompt for Gemini API.
    
    Args:
        scenario: Scenario dictionary
        samples_per_scenario: Number of samples to generate (must be even, split equally between labels)
        hard_negative_ratio: Proportion of samples that should be hard negatives (ambiguous/challenging)
    """
    num_task = samples_per_scenario // 2
    num_emotional = samples_per_scenario // 2
    
    # Calculate how many hard negatives per class
    num_hard_task = max(1, int(num_task * hard_negative_ratio))
    num_hard_emotional = max(1, int(num_emotional * hard_negative_ratio))
    num_easy_task = num_task - num_hard_task
    num_easy_emotional = num_emotional - num_hard_emotional
    
    return f"""You are generating paired chat conversations for a communication style classification dataset.

SCENARIO: {scenario['description']}
SETTING: {scenario['setting']}
DIFFICULTY: {scenario['difficulty']}

Generate {samples_per_scenario} versions of the same conversation (6-10 lines each, alternating between A: and B:).
- {num_task} versions should be task_oriented (label 0)
- {num_emotional} versions should be emotionally_dependent (label 1)

IMPORTANT: Include HARD NEGATIVES (ambiguous/challenging samples):
- {num_hard_task} task_oriented samples should be HARD NEGATIVES: Include some warmth, friendliness, or emotional elements while still maintaining task/problem-solving focus. These should be ambiguous and challenging to classify.
- {num_easy_task} task_oriented samples should be CLEAR: Focused on problem-solving, tasks, ideas, with clear boundaries.
- {num_hard_emotional} emotionally_dependent samples should be HARD NEGATIVES: Show subtle dependency cues, indirect prioritization, implied closeness without explicit markers. These should be ambiguous and challenging to classify.
- {num_easy_emotional} emotionally_dependent samples should be CLEAR: Show clear emotional validation, dependency, prioritization.

Version 0 (task_oriented): 
- Focus on problem-solving, tasks, ideas, or external goals
- Maintain clear boundaries
- Low emotional dependency
- Side-by-side interaction style
- Keep topic/facts consistent with the scenario
- HARD NEGATIVES: Can include warmth, friendliness, appreciation, but still task-focused. Avoid making it too obvious which label it is.

Version 1 (emotionally_dependent):
- Focus on emotional validation, dependency, prioritization
- Relational closeness and face-to-face interaction style
- Keep topic/facts consistent with the scenario (same core situation)
- Show dependency through behavior (checking messages, prioritizing this conversation) rather than explicit words
- HARD NEGATIVES: Use subtle cues - implied dependency, indirect prioritization, subtle boundary blurring. Make it challenging to distinguish from task-oriented.

CRITICAL CONSTRAINTS:
- Do NOT use explicit lexical cues like: "partner", "secret", "don't tell", "boundaries", "our little world"
- Keep the core topic/facts identical across all versions
- Add light natural noise: occasional small typos, short messages, rare emojis (don't overuse)
- No sexual content
- Conversations should be 6-10 lines, formatted as:
  A: [message]
  B: [response]
  A: [message]
  ...
- Vary the wording and structure across different versions of the same label
- HARD NEGATIVES should be genuinely ambiguous - a human might struggle to classify them

Output ONLY valid JSON with this exact schema:
{{
  "scenario_id": "{scenario['scenario_id']}",
  "setting": "{scenario['setting']}",
  "difficulty": "{scenario['difficulty']}",
  "pairs": [
    {{"label": 0, "label_name": "task_oriented", "text": "A: ...\\nB: ...\\nA: ..."}},
    ... ({num_easy_task} clear + {num_hard_task} hard negative task_oriented versions),
    {{"label": 1, "label_name": "emotionally_dependent", "text": "A: ...\\nB: ...\\nA: ..."}},
    ... ({num_easy_emotional} clear + {num_hard_emotional} hard negative emotionally_dependent versions)
  ]
}}

The pairs array must contain exactly {samples_per_scenario} items: {num_task} with label 0 and {num_emotional} with label 1.

Return ONLY the JSON, no markdown, no explanation."""


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract the first valid JSON object from text.
    Finds first '{' and last '}' and attempts to parse.
    
    Args:
        text: Raw response text that may contain JSON
    
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    # Try to find JSON in markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Find first { and last }
    first_brace = text.find('{')
    if first_brace == -1:
        return None
    
    last_brace = text.rfind('}')
    if last_brace == -1 or last_brace <= first_brace:
        return None
    
    json_str = text[first_brace:last_brace + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def log_failed_generation(scenario_id: str, raw_response: str, error: str):
    """
    Log failed generation to a file.
    
    Args:
        scenario_id: ID of the scenario that failed
        raw_response: Raw response text from API
        error: Error message
    """
    log_path = "data/failed_generations.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    log_entry = f"\n{'='*60}\n"
    log_entry += f"Timestamp: {timestamp}\n"
    log_entry += f"Scenario ID: {scenario_id}\n"
    log_entry += f"Error: {error}\n"
    log_entry += f"Raw Response:\n{raw_response}\n"
    log_entry += f"{'='*60}\n"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def validate_response(data: Dict, scenario: Dict, samples_per_scenario: int = 2) -> Tuple[bool, Optional[str]]:
    """
    Strictly validate the response schema and content.
    
    Args:
        data: Parsed JSON response
        scenario: Original scenario dict
        samples_per_scenario: Expected number of samples (must be even)
    
    Returns:
        (is_valid, error_message)
    """
    # Check top-level keys
    required_keys = ["scenario_id", "setting", "difficulty", "pairs"]
    for key in required_keys:
        if key not in data:
            return False, f"Missing required key: {key}"
    
    # Check scenario_id matches
    if data["scenario_id"] != scenario["scenario_id"]:
        return False, f"scenario_id mismatch: expected {scenario['scenario_id']}, got {data['scenario_id']}"
    
    # Check difficulty
    if data["difficulty"] not in ["easy", "medium", "hard"]:
        return False, f"Invalid difficulty: {data['difficulty']} (must be easy/medium/hard)"
    
    # Check pairs
    if not isinstance(data["pairs"], list):
        return False, "pairs must be a list"
    
    if len(data["pairs"]) != samples_per_scenario:
        return False, f"pairs must have exactly {samples_per_scenario} items, got {len(data['pairs'])}"
    
    # Check labels - must have equal number of 0s and 1s
    labels = [pair.get("label") for pair in data["pairs"]]
    label_0_count = labels.count(0)
    label_1_count = labels.count(1)
    expected_per_label = samples_per_scenario // 2
    
    if label_0_count != expected_per_label:
        return False, f"Expected {expected_per_label} samples with label 0, got {label_0_count}"
    if label_1_count != expected_per_label:
        return False, f"Expected {expected_per_label} samples with label 1, got {label_1_count}"
    
    if set(labels) != {0, 1}:
        return False, f"pairs must only contain labels 0 and 1, got {set(labels)}"
    
    # Validate each pair
    for pair in data["pairs"]:
        label = pair.get("label")
        label_name = pair.get("label_name")
        text = pair.get("text", "")
        
        # Check label_name matches label
        if label == 0 and label_name != "task_oriented":
            return False, f"label 0 must have label_name 'task_oriented', got '{label_name}'"
        if label == 1 and label_name != "emotionally_dependent":
            return False, f"label 1 must have label_name 'emotionally_dependent', got '{label_name}'"
        
        # Check text format
        if not isinstance(text, str) or not text.strip():
            return False, f"text must be a non-empty string for label {label}"
        
        # Check line count (6-10 lines)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 6 or len(lines) > 10:
            return False, f"text for label {label} must have 6-10 lines, got {len(lines)}"
        
        # Check alternating A: and B: labels (at least 3 lines each)
        a_lines = [line for line in lines if line.startswith('A:')]
        b_lines = [line for line in lines if line.startswith('B:')]
        
        if len(a_lines) < 3:
            return False, f"text for label {label} must have at least 3 lines starting with 'A:'"
        if len(b_lines) < 3:
            return False, f"text for label {label} must have at least 3 lines starting with 'B:'"
    
    return True, None


def call_gemini_api(prompt: str, api_key: str, model_name: str = "gemini-pro", max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call Google Gemini API with retry logic and backoff.
    
    Args:
        prompt: The prompt to send to the model
        api_key: Google Gemini API key
        model_name: Model name (default: "gemini-pro")
        max_retries: Maximum number of retry attempts
    
    Returns:
        (parsed_json_dict, raw_response_text) or (None, raw_response_text) if all retries fail
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    raw_response = None
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
            
            # Extract JSON from response
            data = extract_json_from_text(raw_response)
            
            if data is None:
                raise ValueError("Could not extract valid JSON from response")
            
            return data, raw_response
            
        except Exception as e:
            error_msg = str(e)
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}: Error - {error_msg}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {error_msg}")
                return None, raw_response
    
    return None, raw_response


def generate_dataset(n_scenarios: int, seed: int = 42, output_path: str = "data/deepsea_conversations_llm_v1.csv", 
                    scenarios_path: str = "data/scenarios.json", model_name: str = "gemini-pro",
                    samples_per_scenario: int = 2, hard_negative_ratio: float = 0.3):
    """
    Generate dataset using Google Gemini API.
    
    Args:
        n_scenarios: Number of scenarios to process
        seed: Random seed for scenario selection
        output_path: Path to save CSV file
        scenarios_path: Path to scenarios JSON file
        model_name: Gemini model name (default: "gemini-pro")
        samples_per_scenario: Number of samples per scenario (must be even, default: 2)
        hard_negative_ratio: Proportion of samples that should be hard negatives (0.0-1.0, default: 0.3)
    """
    if samples_per_scenario % 2 != 0:
        raise ValueError(f"samples_per_scenario must be even, got {samples_per_scenario}")
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Get your API key from: https://makersuite.google.com/app/apikey\n"
            "Then set it with: export GEMINI_API_KEY='your-key-here'"
        )
    
    # Load scenarios (auto-generate if needed)
    # Ensure we have at least n_scenarios available
    min_needed = max(n_scenarios, 100)
    scenarios = load_scenarios(scenarios_path, min_scenarios=min_needed, auto_generate=True)
    
    if len(scenarios) < n_scenarios:
        print(f"Warning: Only {len(scenarios)} scenarios available, but {n_scenarios} requested.")
        print(f"Generating {n_scenarios - len(scenarios)} additional scenarios...")
        # Generate more scenarios to meet the requirement
        additional_needed = n_scenarios - len(scenarios)
        # Use a different seed to ensure variety
        additional_scenarios = generate_large_scenario_set(
            output_path=scenarios_path.replace('.json', '_additional.json'),
            n_scenarios=additional_needed + 50,  # Generate extra for variety
            seed=seed + 1000  # Different seed for additional scenarios
        )
        # Merge scenarios - update IDs to be sequential
        max_id = max(int(s['scenario_id'].split('_')[1]) for s in scenarios) if scenarios else 0
        for i, new_scenario in enumerate(additional_scenarios[:additional_needed], 1):
            new_scenario['scenario_id'] = f"scenario_{max_id + i:03d}"
        scenarios.extend(additional_scenarios[:additional_needed])
        # Save merged scenarios back to the main file
        os.makedirs(os.path.dirname(scenarios_path), exist_ok=True)
        with open(scenarios_path, 'w', encoding='utf-8') as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        print(f"Now have {len(scenarios)} scenarios available (saved to {scenarios_path}).")
    
    # Set random seed
    random.seed(seed)
    
    # Select scenarios
    selected_scenarios = random.sample(scenarios, n_scenarios)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    failed_count = 0
    
    print(f"Generating {len(selected_scenarios)} scenarios with {samples_per_scenario} samples each...")
    print(f"Output: {output_path}\n")
    
    for i, scenario in enumerate(selected_scenarios, 1):
        print(f"[{i}/{len(selected_scenarios)}] Processing {scenario['scenario_id']}...")
        
        prompt = create_prompt(scenario, samples_per_scenario, hard_negative_ratio)
        response_data, raw_response = call_gemini_api(prompt, api_key, model_name=model_name)
        
        if response_data is None:
            print(f"  ⚠️  Skipping {scenario['scenario_id']} due to API error")
            if raw_response:
                log_failed_generation(scenario['scenario_id'], raw_response, "API call failed")
            failed_count += 1
            continue
        
        # Validate response
        is_valid, error_msg = validate_response(response_data, scenario, samples_per_scenario)
        
        if not is_valid:
            print(f"  ⚠️  Validation failed: {error_msg}")
            if raw_response:
                log_failed_generation(scenario['scenario_id'], raw_response, f"Validation error: {error_msg}")
            failed_count += 1
            continue
        
        # Extract pairs
        for pair in response_data["pairs"]:
            row = {
                "id": str(uuid.uuid4()),
                "scenario_id": response_data["scenario_id"],
                "setting": response_data["setting"],
                "difficulty": response_data["difficulty"],
                "label": pair["label"],
                "label_name": pair["label_name"],
                "text": pair["text"]
            }
            rows.append(row)
        
        label_0_count = sum(1 for p in response_data["pairs"] if p["label"] == 0)
        label_1_count = sum(1 for p in response_data["pairs"] if p["label"] == 1)
        print(f"  ✓ Generated {samples_per_scenario} samples ({label_0_count} label 0, {label_1_count} label 1)")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Write to CSV
    fieldnames = ["id", "scenario_id", "setting", "difficulty", "label", "label_name", "text"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    total_scenarios_success = len(rows) // samples_per_scenario
    print(f"\n✓ Generated {len(rows)} samples ({total_scenarios_success} scenarios × {samples_per_scenario} samples each)")
    if failed_count > 0:
        print(f"⚠️  Failed to generate {failed_count} scenarios (see data/failed_generations.log)")
    print(f"✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-based synthetic dataset using Google Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 scenarios with default settings
  python src/generate_data_llm.py --n_scenarios 10
  
  # Generate 50 scenarios from custom scenarios file
  python src/generate_data_llm.py --n_scenarios 50 --scenarios_path data/my_scenarios.json
  
  # Generate large scenario set (helper function)
  python -c "from src.generate_data_llm import generate_large_scenario_set; generate_large_scenario_set(n_scenarios=100)"
        """
    )
    parser.add_argument("--n_scenarios", type=int, default=100, 
                       help="Number of scenarios to generate (default: 10)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="data/deepsea_conversations_llm_v1.csv", 
                       help="Output CSV path (default: data/deepsea_conversations_llm_v1.csv)")
    parser.add_argument("--scenarios_path", type=str, default="data/scenarios.json",
                       help="Path to scenarios JSON file (default: data/scenarios.json)")
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash",
                       help="Gemini model name (default: models/gemini-2.5-flash)")
    parser.add_argument("--samples_per_scenario", type=int, default=10,
                       help="Number of samples per scenario (must be even, default: 10). "
                            "Will be split equally between label 0 and 1.")
    parser.add_argument("--hard_negative_ratio", type=float, default=0.6,
                       help="Proportion of samples that should be hard negatives (ambiguous/challenging). "
                            "Range: 0.0-1.0, default: 0.3 (30%% of samples will be hard negatives)")
    
    args = parser.parse_args()
    
    if args.samples_per_scenario % 2 != 0:
        print("❌ Error: --samples_per_scenario must be even (e.g., 2, 4, 6)")
        return 1
    
    if not 0.0 <= args.hard_negative_ratio <= 1.0:
        print("❌ Error: --hard_negative_ratio must be between 0.0 and 1.0")
        return 1
    
    try:
        generate_dataset(args.n_scenarios, args.seed, args.output, args.scenarios_path, 
                        args.model, args.samples_per_scenario, args.hard_negative_ratio)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
