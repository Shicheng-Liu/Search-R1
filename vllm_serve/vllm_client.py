#!/usr/bin/env python3
"""
VLLM Client for Search-Enabled Reasoning

A client for interacting with VLLM servers to perform search-enabled reasoning
and answer evaluation with scoring capabilities.
"""

import argparse
import json
import logging
import re
from typing import List, Tuple, Optional

from openai import OpenAI


DEFAULT_DATA_PATH = "./outputs/log_val_traj/val-search-r1-ppo-qwen2.5-7b-em-gae_20250814_043740/trajectories_val_batch_0.json"


# ============================================================================
# VLLM Client Class
# ============================================================================

class VLLMClient:
    """Client for interacting with VLLM servers."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002, model: str = "openai/gpt-oss-20b"):
        """Initialize VLLM client.
        
        Args:
            host: Host address for VLLM server
            port: Port number for VLLM server
            model: Model name to use
        """
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
        self.logger = logging.getLogger(__name__)
    
    def generate_text(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Generate text using the VLLM server.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response or None if error occurred
        """
        try:
            self.logger.info(f"Connecting to VLLM server at: {self.base_url}")
            
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            
            return chat_response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return None


# ============================================================================
# Judge and Evaluation Classes
# ============================================================================

class JudgeEvaluator:
    """Handles judge prompt creation and response evaluation."""
    
    @staticmethod
    def create_turn_judge_prompt(prompt: str, turns: List[str], ground_truth: str) -> str:
        """Create evaluation prompt for prompt-response assessment.
        
        Args:
            prompt: Original prompt
            turns: Pre-divided turn texts
            ground_truth: Ground truth answer for comparison
            
        Returns:
            Formatted judge prompt
        """
        
        prompt_text = f"PROMPT:\n{prompt}\n"
        turns_text = ""
        for i, turn in enumerate(turns, 1):
            turns_text += f"TURN {i}:\n{turn}\n"
        
        ground_truth_text = f"GROUND TRUTH:\n{ground_truth}\n"
        
        judge_prompt = f"""
You are an expert evaluator for multi-turn search-augmented reasoning systems. Given a user prompt, ground truth answer, and multi-turn generated response, evaluate each turn's effectiveness and compliance.

## EVALUATION TASK

Assess each turn's format compliance, content quality, and contribution toward the ground truth answer.

## SCORING CRITERIA

### FINAL TURN (Last Turn) - Score Range: [-1.0 to 1.0]

**Format Compliance:**
• Required: `<think>...</think><answer>...</answer>` (tags only, once each, in order)
• Answer in `<answer>` tag must not exceed 5 tokens

**Answer Correctness:**
• Correct and complete answer in `<answer>` tag that matches the ground truth

**Scoring Rules:**
• If format is incorrect: Final Turn Score = -1.0
• If format is correct, answer is incorrect: Final Turn Score = 0.2
• If format is correct, answer is correct: Final Turn Score = 1.0

### INTERMEDIATE TURNS - Score Range: [-1.0 to 1.0]

**Format Compliance:**
• Required: `<think>...</think><search>...</search><information>...</information>` (tags only, once each, in order)
• Correct format: +0.1
• Incorrect format: -0.2

**Information Quality:**
• Relevant information in `<information>` tag that helps toward the ground truth answer (e.g., ground truth exists in the retrieved result within `<information>` tag): +0.3
• Irrelevant or unhelpful information in `<information>` tag: +0.0

**Search Efficiency Penalty:**
• Number of searches = Total count of `<search>` tags across all turns from Turn 1 up to and including the current turn
• Search penalty = Number of searches × (-0.1)
• Encourages finding answers with fewer searches

**Intermediate Turn Score = Format Compliance + Information Quality + Search Penalty**

## OUTPUT FORMAT

Provide your evaluation using ONLY these XML tags:

<reasoning>
Systematically evaluate each turn: check format compliance, assess content quality, calculate scores with clear explanations
</reasoning>

<score>
Turn1: X.X
Turn2: X.X
Turn3: X.X
...
</score>

⚠️ REQUIREMENTS:
• Must provide exactly {len(turns)} scores (one per turn)
• Use decimal format (e.g., 0.5, -0.3, 1.0)
• Use only the specified XML tags, no additional text

## EVALUATION DATA

{prompt_text}
{turns_text}
{ground_truth_text}
**TURNS TO EVALUATE: {len(turns)}**

## Your Evaluation
"""




#         judge_prompt = f"""
# You are an expert evaluator for multi-turn search-augmented reasoning systems. Given a user prompt, ground truth answer, and multi-turn generated response, evaluate each turn's effectiveness, compliance, and progress toward the ground truth.

# ## EVALUATION TASK

# Assess each turn's format compliance, content quality, and contribution toward the ground truth answer.

# ## SCORING CRITERIA

# ### FINAL TURN (Last Turn) - Score Range: [-1.0 to 1.0]

# **Format Compliance:**
# • Required: `<think>...</think><answer>...</answer>` (tags only, once each, in order)
# • Answer in `<answer>` tag must not exceed 5 tokens

# **Answer Correctness:**
# • Correct and complete answer in `<answer>` tag that matches the ground truth

# **Scoring Rules:**
# • If format is incorrect: Final Turn Score = -1.0
# • If format is correct, answer is incorrect: Final Turn Score = 0.2
# • If format is correct, answer is correct: Final Turn Score = 1.0

# ### INTERMEDIATE TURNS - Score Range: [-1.0 to 1.0]

# **Format Compliance:**
# • Required: `<think>...</think><search>...</search><information>...</information>` (tags only, once each, in order)
# • Correct format: +0.1
# • Incorrect format: -0.2

# **Distance Reward:**
# • Must be in **[-1.0, 1.0]**
# • **Distance Reward = 1.0 if and only if** the information in `<information>` tag contains the correct ground-truth answer
# • The distance reward should be higher (but <= 1.0) if the information in `<information>` tag is semantically closer to the ground truth (more key facts correct, fewer errors, less ambiguity)
# • The distance reward should be lower (but >= -1.0) if the information in `<information>` tag is wrong, contradictory, or misleading with respect to the ground truth
# • The distance reward is 0.0 if the information in `<information>` tag is irrelevant or unhelpful

# **Intermediate Turn Score = Format Compliance + Distance Reward**

# ## OUTPUT FORMAT

# Provide your evaluation using ONLY these XML tags:

# <reasoning>
# Systematically evaluate each turn: check format compliance, assess content quality, calculate scores with clear explanations
# </reasoning>

# <score>
# Turn1: X.X
# Turn2: X.X
# Turn3: X.X
# ...
# </score>

# ⚠️ REQUIREMENTS:
# • Must provide exactly {len(turns)} scores (one per turn)
# • Use decimal format (e.g., 0.5, -0.3, 1.0)
# • Use only the specified XML tags, no additional text

# ## EVALUATION DATA

# {prompt_text}
# {turns_text}
# {ground_truth_text}
# **TURNS TO EVALUATE: {len(turns)}**

# ## Your Evaluation
# """


        return judge_prompt

    @staticmethod
    def extract_turn_scores_from_judge_response(judge_response: str, num_turns: int) -> List[float]:
        """Extract individual turn scores from judge response.
        
        Args:
            judge_response: The judge's evaluation response
            num_turns: Expected number of turns
            
        Returns:
            List of scores for each turn
        """
        scores = []
        try:
            # Extract score section from the result
            score_pattern = r'<score>(.*?)</score>'
            score_match = re.search(score_pattern, judge_response, re.DOTALL)
            
            if score_match:
                score_text = score_match.group(1).strip()
                # Parse individual turn scores
                turn_pattern = r'Turn(\d+):\s*([-+]?\d*\.?\d+)'
                turn_matches = re.findall(turn_pattern, score_text)
                
                for _, score_str in turn_matches:
                    score = float(score_str)
                    scores.append(score)
            else:
                print(f"Warning: No score section found in judge response, defaulting to zero scores")
                scores = [0.0] * num_turns
        
        except Exception as e:
            print(f"Error parsing scores: {e}, defaulting to zero scores")
            scores = [0.0] * num_turns
        
        # Ensure we have the correct number of scores for number of turns
        if len(scores) != num_turns:
            print(f"\nWarning: Expected {num_turns} scores, got {len(scores)}. Scores: {scores}. Padding.")
            print(f"Judge response: {judge_response}")
            if len(scores) < num_turns:
                print(f"Padding with zeros.")              
                scores.extend([0.0] * (num_turns - len(scores)))
            else:
                print(f"Truncating extra scores.")
                scores = scores[:num_turns]
        
        return scores

# class JudgeEvaluator:
#     """Handles judge prompt creation and response evaluation."""
#     @staticmethod
#     def create_turn_judge_prompt(prompt: str, turns: List[str], ground_truth: str) -> str:
#         """Create evaluation prompt for distance-to-ground-truth per turn.

#         Args:
#             prompt: Original user prompt
#             turns: Pre-divided turn texts (TURN 1..N)
#             ground_truth: Final short answer (ground truth)

#         Returns:
#             Judge prompt string
#         """
#         prompt_text = f"PROMPT:\n{prompt}\n"

#         turns_text = ""
#         for i, turn in enumerate(turns, 1):
#             turns_text += f"TURN {i}:\n{turn}\n"

#         ground_truth_text = f"GROUND TRUTH FINAL SHORT ANSWER:\n{ground_truth}\n"

#         num_turns = len(turns)

#         judge_prompt = f"""
# You are an expert evaluator for multi-turn search-augmented reasoning systems.

# You are given:
# - USER PROMPT
# - GROUND TRUTH FINAL SHORT ANSWER
# - A multi-turn generated response (TURN 1..N)

# Your task: for each turn t, estimate how far the system is (after reading turns 1..t)
# from being able to output the correct final SHORT answer.

# ========================
# DISTANCE-TO-ANSWER
# ========================

# After each turn t:
# 1) Infer the system's best current SHORT answer guess (even if not explicitly stated).
# 2) Assign a DISTANCE score D_t in [0.0, 1.0] indicating how far this best guess is from the ground truth.

# Interpretation:
# - D_t = 0.0: correct final answer is fully determined (best guess matches ground truth meaning).
# - D_t = 1.0: very far (no useful evidence or clearly wrong direction).

# Distance should reflect:
# - semantic correctness of the best guess vs ground truth
# - whether the answer is uniquely determined vs still uncertain/ambiguous
# - penalize wrong committed claims that would mislead future turns (distance can INCREASE)

# Distance anchors:
# - 0.0: exactly correct meaning (case/format differences ok if meaning identical)
# - 0.1–0.3: almost correct; minor detail/wording mismatch only
# - 0.4–0.6: partially correct or ambiguous; missing a key qualifier/constraint
# - 0.7–0.9: weak guess; only broad direction; major uncertainty remains
# - 1.0: wrong / contradictory / no relevant progress

# IMPORTANT:
# - Do NOT score based on style, verbosity, or confidence alone.
# - If a turn is only formatting or generic reasoning without new evidence, distance should not decrease.
# - If a turn introduces a confident wrong claim, distance should increase (or remain high).

# ========================
# OUTPUT FORMAT (STRICT)
# ========================

# Return ONLY these XML tags, with no additional text:

# <reasoning>
# Turn-by-turn: (a) best short answer guess so far, (b) why distance increased/decreased.
# </reasoning>

# <score>
# Turn1: D.D
# Turn2: D.D
# ...
# Turn{num_turns}: D.D
# </score>

# <best_answer>
# Turn1: ...
# Turn2: ...
# ...
# Turn{num_turns}: ...
# </best_answer>

# REQUIREMENTS:
# - Provide exactly {num_turns} scores (one per turn) in <score>
# - Each score must be a decimal in [0.0, 1.0]
# - Provide exactly {num_turns} best answers (one per turn) in <best_answer>
# - No extra tags, no extra prose outside the XML tags

# ========================
# EVALUATION DATA
# ========================

# {prompt_text}
# {turns_text}
# {ground_truth_text}

# ## Your Evaluation
# """.strip()

#         return judge_prompt

#     @staticmethod
#     def extract_turn_scores_from_judge_response(judge_response: str, num_turns: int) -> List[float]:
#         """Parse judge distances and return per-turn distance-based rewards (progress).

#         The judge is expected to output D_t (distance) in <score>:
#             Turn1: D1
#             Turn2: D2
#             ...
#         We convert to rewards:
#             r1 = 1.0 - D1
#             rt = D_{t-1} - D_t  (t>=2)

#         Args:
#             judge_response: Judge's raw response text
#             num_turns: Expected number of turns

#         Returns:
#             List[float] length `num_turns`, containing progress rewards.
#         """

#         def _clamp01(x: float) -> float:
#             if x < 0.0:
#                 return 0.0
#             if x > 1.0:
#                 return 1.0
#             return x

#         def _distances_to_rewards(distances: List[float], d0: float = 1.0) -> List[float]:
#             rewards: List[float] = []
#             prev = d0
#             for d in distances:
#                 rewards.append(prev - d)
#                 prev = d
#             return rewards

#         # 1) Parse distances
#         distances: List[float] = []
#         try:
#             score_match = re.search(r"<score>(.*?)</score>", judge_response, re.DOTALL | re.IGNORECASE)
#             if not score_match:
#                 # conservative: no parse => no progress everywhere
#                 # distances all 1.0 -> rewards all 0.0
#                 return [0.0] * num_turns

#             score_text = score_match.group(1).strip()

#             # Allow "Turn 1: 0.7" or "Turn1:0.7"
#             turn_matches = re.findall(
#                 r"Turn\s*(\d+)\s*:\s*([-+]?\d*\.?\d+)",
#                 score_text,
#                 flags=re.IGNORECASE,
#             )
#             if not turn_matches:
#                 return [0.0] * num_turns

#             by_turn = {}
#             for turn_idx_str, score_str in turn_matches:
#                 try:
#                     t = int(turn_idx_str)
#                     d = _clamp01(float(score_str))
#                     by_turn[t] = d
#                 except Exception:
#                     continue

#             # Build Turn1..TurnN distances; pad missing turns with last known distance (or 1.0)
#             last_d = 1.0
#             for t in range(1, num_turns + 1):
#                 if t in by_turn:
#                     last_d = by_turn[t]
#                     distances.append(last_d)
#                 else:
#                     distances.append(last_d)

#         except Exception:
#             return [0.0] * num_turns

#         # Guard length
#         if len(distances) != num_turns:
#             if len(distances) < num_turns:
#                 pad_val = distances[-1] if distances else 1.0
#                 distances.extend([pad_val] * (num_turns - len(distances)))
#             else:
#                 distances = distances[:num_turns]

#         # 2) Convert distances -> progress rewards
#         rewards = _distances_to_rewards(distances, d0=1.0)

#         # Optional: clamp rewards to a sane range, if you want stability
#         # (progress rewards are naturally in [-1, 1] given distances in [0,1])
#         rewards = [max(-1.0, min(1.0, r)) for r in rewards]

#         return rewards


    @staticmethod
    def create_outcome_judge_prompt(prompt: str, turns: List[str], ground_truth: str) -> str:
        """Create evaluation prompt for prompt-response assessment.
        
        Args:
            prompt: Original prompt
            turns: Pre-divided turn texts
            ground_truth: Ground truth answer for comparison
            
        Returns:
            Formatted judge prompt
        """
        
        prompt_text = f"PROMPT:\n{prompt}\n"
        turns_text = ""
        for i, turn in enumerate(turns, 1):
            turns_text += f"TURN {i}:\n{turn}\n"
        
        ground_truth_text = f"GROUND TRUTH:\n{ground_truth}\n"
        
        judge_prompt = f"""
You are an expert evaluator for multi-turn search-augmented reasoning systems. Given a user prompt, ground truth answer, and multi-turn generated response, determine whether the final answer matches the ground truth.

## EVALUATION TASK

Evaluate whether the multi-turn response provides a correct final answer that matches the ground truth.

## SCORING CRITERIA

**Score 1.0 (Correct):**
• The answer within `<answer></answer>` tags matches the ground truth

**Score 0.0 (Incorrect):**
• No `<answer></answer>` tags found, OR
• The answer within `<answer></answer>` tags does not match the ground truth, OR
• The answer in `<answer>` tag exceeds 5 tokens

## OUTPUT FORMAT

Provide your evaluation using this format:

<reasoning>
[Your step-by-step reasoning about whether the answer matches the ground truth]
</reasoning>

<score>
1.0 or 0.0
</score>

⚠️ REQUIREMENTS:
• First provide reasoning, then the score
• Score must be exactly 1.0 or 0.0

## EVALUATION DATA

{prompt_text}
{turns_text}
{ground_truth_text}

## Your Evaluation
"""
        return judge_prompt

    @staticmethod
    def extract_outcome_score_from_judge_response(judge_response: str) -> float:
        """Extract outcome score from judge response.
        
        Args:
            judge_response: The judge's evaluation response
            
        Returns:
            Single outcome score as float
        """
        try:
            # Extract score from XML tag
            score_pattern = r'<score>(.*?)</score>'
            match = re.search(score_pattern, judge_response, re.DOTALL)
            
            if match:
                score_text = match.group(1).strip()
                score = float(score_text)
                # Clamp score to valid range
                return max(0.0, min(1.0, score))
            else:
                print(f"Warning: Could not find score tag in judge response")
                return 0.0
                
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse outcome score: {e}")
            return 0.0


# ============================================================================
# Data Processing Classes
# ============================================================================

class DataProcessor:
    """Handles data loading and processing operations."""
    
    @staticmethod
    def extract_prompt_from_chat_format(text: str) -> str:
        """Extract the user prompt from chat format.
        
        Args:
            text: Chat format text containing <|im_start|>user and <|im_end|> tags
            
        Returns:
            Extracted prompt text
        """
        # Find content between <|im_start|>user and <|im_end|>
        pattern = r'<\|im_start\|>user\s*\n(.*?)\n<\|im_end\|>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return text  # Return original text if no pattern found


    @staticmethod
    def get_sample_data(num_samples: int = 5, json_file: str = DEFAULT_DATA_PATH) -> List[Tuple[str, List[str], str]]:
        """Get multiple sample prompts and turn texts from JSON file.
        
        Args:
            num_samples: Number of samples to retrieve
            json_file: Path to JSON file containing samples
            
        Returns:
            List of (prompt, turn_texts, ground_truth) tuples
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        # Handle -1 as "all samples"
        sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))
        for i in range(sample_count):
            sample = data[i]
            raw_prompt = sample["prompt"]
            
            # Extract the actual user prompt from chat format
            prompt = DataProcessor.extract_prompt_from_chat_format(raw_prompt)
            turn_texts = sample["turn_texts"]
            ground_truth_list = sample["ground_truth"]
            ground_truth = ", ".join(ground_truth_list) if ground_truth_list else None
            samples.append((prompt, turn_texts, ground_truth))
        
        return samples


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VLLM Client for Search-Enabled Reasoning")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--judge-mode", type=str, default="outcome", 
                        choices=["turn", "outcome"],
                        help="Judge evaluation mode: turn (evaluate each turn), outcome (evaluate final result)")
    return parser.parse_args()


def main():
    """Main function to run the client evaluation."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    

    # Initialize components with parsed arguments
    samples = DataProcessor.get_sample_data()
    client = VLLMClient(host=args.host, port=args.port, model=args.model)
    judge_evaluator = JudgeEvaluator()
    
    # Evaluate samples
    for i, (sample_prompt, sample_turns, ground_truth) in enumerate(samples):
        print(f"\nSAMPLE {i+1}:")
        print("-" * 100)
        
        if args.judge_mode == "outcome":
            judge_prompt = judge_evaluator.create_outcome_judge_prompt(sample_prompt, sample_turns, ground_truth)
        elif args.judge_mode == "turn":
            judge_prompt = judge_evaluator.create_turn_judge_prompt(sample_prompt, sample_turns, ground_truth)
        
        print(f"Evaluating sample {i+1}...")
        result = client.generate_text(judge_prompt)
        
        print("=== JUDGE RESPONSE DEBUG ===")
        print(judge_prompt)
        print(result)
        print("=== END JUDGE RESPONSE ===")
        
        if result:
            if args.judge_mode == "outcome":
                score = judge_evaluator.extract_outcome_score_from_judge_response(result)
                print(f"Sample {i+1} score: {score}")
            elif args.judge_mode == "turn":
                scores = judge_evaluator.extract_turn_scores_from_judge_response(result, len(sample_turns))
                print(f"Sample {i+1} scores: {scores}")
        else:
            print(f"Failed to evaluate sample {i+1}")
        
        print("-" * 100)


if __name__ == "__main__":
    main()