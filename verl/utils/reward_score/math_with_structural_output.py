from typing import Tuple, Optional
from .math import compute_score as math_compute_score
# from math_verify import LatexExtractionConfig, parse, verify
# from latex2sympy2_extended import NormalizationConfig
import re


def compute_score(solution_str: str, ground_truth: str) -> float:
    print("\n" + "="*80)
    print(f" Origin solution: {solution_str}\n")
    print(" Processing New Sample ".center(80, '='))
    retval = 0.
    try:
        answer_str, response_str = extract_solution(solution_str)
        print(f"\n[Model Response]\n{response_str}\n")
        if answer_str is None:
            acc_score = 0
        else:
            acc_score = math_compute_score(answer_str, ground_truth)
        print(f"\n[Ground Truth] {ground_truth}\n")
        print(f"\n  Accuracy validation: {'PASS' if acc_score else 'FAIL'}")
        format_score = format_reward(response_str)
        print(f"\n Format validation: {'PASS' if format_score else 'FAIL'}")
        retval = acc_score + format_score
    except Exception as e:
        print(e)

    return retval


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


# def accuracy_reward(solution_str, ground_str):
#     ground_str_parsed = parse(
#         ground_str,
#         extraction_mode="first_match",
#         extraction_config=[LatexExtractionConfig()],
#     )
#     if ground_str_parsed == 0:
#         reward = 1.0
#         print(f"Failed to parse ground solution: {ground_str}")
#     else:
#         answer_parsed = parse(
#             solution_str,
#             extraction_config=[
#                 LatexExtractionConfig(
#                     normalization_config=NormalizationConfig(
#                         nits=False,
#                         malformed_operators=False,
#                         basic_latex=True,
#                         equations=True,
#                         boxed="all",
#                         units=True,
#                     ),
#                     # Ensures that boxed is tried first
#                     boxed_match_priority=0,
#                     try_extract_without_anchor=False,
#                 )
#             ],
#             extraction_mode="first_match",
#         )
#         # Reward 1 if the content is the same as the ground truth, 0 otherwise
#         reward = float(verify(answer_parsed, ground_str_parsed))
#     return reward


def format_reward(processed_str: str) -> float:
    valid_response = validate_response_structure(processed_str)
    if valid_response:
        return 1.0
    return 0.0


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed
