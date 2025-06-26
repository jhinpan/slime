"""
Reward utilities for kernel code generation tasks.

This module provides functions to evaluate the quality of generated CUDA kernel code.
"""

import re
import ast
from typing import Dict, List, Tuple, Optional


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from the generated text."""
    code_blocks = []
    
    # Try to find code blocks with ``` markers
    pattern = r'```(?:python|cuda|cpp|c\+\+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        code_blocks.extend(matches)
    
    # If no code blocks found, try to extract code that looks like Python
    if not code_blocks and text:
        # Look for class definitions
        class_pattern = r'(class\s+\w+.*?(?=class\s+\w+|$))'
        class_matches = re.findall(class_pattern, text, re.DOTALL)
        if class_matches:
            code_blocks.extend(class_matches)
    
    return code_blocks


def check_cuda_kernel_syntax(code: str) -> Tuple[bool, List[str]]:
    """Check if the code contains valid CUDA kernel syntax."""
    errors = []
    
    # Check for CUDA kernel markers
    has_cuda_kernel = "__global__" in code or "__device__" in code
    has_kernel_launch = "<<<" in code and ">>>" in code
    
    if not has_cuda_kernel and not has_kernel_launch:
        errors.append("No CUDA kernel markers found")
    
    # Check for load_inline usage (required for PyTorch CUDA extension)
    has_load_inline = "load_inline" in code
    if not has_load_inline:
        errors.append("No load_inline usage found for PyTorch CUDA extension")
    
    # Check for ModelNew class
    has_model_new = "class ModelNew" in code
    if not has_model_new:
        errors.append("ModelNew class not found")
    
    return len(errors) == 0, errors


def check_python_syntax(code: str) -> Tuple[bool, List[str]]:
    """Check if the Python parts of the code are syntactically valid."""
    errors = []
    
    # Extract Python code (outside of C++ strings)
    python_code = code
    
    # Remove C++ code strings to avoid syntax errors
    cpp_pattern = r'""".*?"""'
    python_code = re.sub(cpp_pattern, '""""""', python_code, flags=re.DOTALL)
    
    try:
        ast.parse(python_code)
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e}")
    
    return len(errors) == 0, errors


def check_required_functions(code: str) -> Tuple[bool, List[str]]:
    """Check if the code contains required functions."""
    errors = []
    
    # Check for required functions in the generated code
    required_patterns = [
        (r'def\s+forward\s*\(', "forward method"),
        (r'def\s+__init__\s*\(', "__init__ method"),
    ]
    
    for pattern, name in required_patterns:
        if not re.search(pattern, code):
            errors.append(f"Missing {name}")
    
    return len(errors) == 0, errors


def check_cuda_completeness(code: str) -> Tuple[bool, List[str]]:
    """Check if the CUDA implementation is complete."""
    errors = []
    
    # Check for kernel implementation
    kernel_pattern = r'__global__\s+void\s+\w+\s*\([^)]*\)\s*\{'
    if not re.search(kernel_pattern, code):
        errors.append("No complete CUDA kernel implementation found")
    
    # Check for kernel launch
    if "<<<" in code and ">>>" in code:
        # Check if kernel launch has proper parameters
        launch_pattern = r'<<<[^,]+,[^>]+>>>'
        if not re.search(launch_pattern, code):
            errors.append("Invalid kernel launch syntax")
    
    return len(errors) == 0, errors


def evaluate_kernel_quality(code: str) -> float:
    """
    Evaluate the quality of generated kernel code.
    Returns a score between 0 and 1.
    """
    score = 0.0
    max_score = 0.0
    
    # Check CUDA kernel syntax (40 points)
    cuda_valid, cuda_errors = check_cuda_kernel_syntax(code)
    max_score += 40
    if cuda_valid:
        score += 40
    else:
        # Partial credit based on errors
        score += max(0, 40 - len(cuda_errors) * 10)
    
    # Check Python syntax (20 points)
    python_valid, python_errors = check_python_syntax(code)
    max_score += 20
    if python_valid:
        score += 20
    
    # Check required functions (20 points)
    functions_valid, function_errors = check_required_functions(code)
    max_score += 20
    if functions_valid:
        score += 20
    else:
        # Partial credit
        score += max(0, 20 - len(function_errors) * 10)
    
    # Check CUDA completeness (20 points)
    cuda_complete, completeness_errors = check_cuda_completeness(code)
    max_score += 20
    if cuda_complete:
        score += 20
    else:
        # Partial credit
        score += max(0, 20 - len(completeness_errors) * 10)
    
    # Additional bonus for optimization techniques (up to 10 bonus points)
    optimization_patterns = [
        (r'__shared__', 2),  # Shared memory usage
        (r'gridDim|blockDim|blockIdx|threadIdx', 2),  # Proper thread indexing
        (r'__syncthreads\(\)', 2),  # Thread synchronization
        (r'#pragma unroll', 2),  # Loop unrolling
        (r'__restrict__', 2),  # Restrict keyword
    ]
    
    for pattern, points in optimization_patterns:
        if re.search(pattern, code):
            score += points
            max_score += points
    
    # Normalize to 0-1 range
    return min(1.0, max(0.0, score / max_score))


def get_kernel_code_reward(item: Dict) -> float:
    """
    Calculate reward for kernel code generation task.
    
    Args:
        item: Dictionary containing the generated response
        
    Returns:
        float: Reward value between 0 and 1
    """
    try:
        # Extract the assistant's response
        messages = item.get("messages", [])
        response = ""
        
        for msg in messages:
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                break
        
        if not response:
            return 0.0
        
        # Extract code blocks
        code_blocks = extract_code_blocks(response)
        
        if not code_blocks:
            # If no code blocks found, try to evaluate the entire response
            return evaluate_kernel_quality(response) * 0.5  # Penalty for no proper formatting
        
        # Evaluate each code block and take the best score
        best_score = 0.0
        for code in code_blocks:
            score = evaluate_kernel_quality(code)
            best_score = max(best_score, score)
        
        return best_score
        
    except Exception as e:
        print(f"Error in reward calculation: {e}")
        return 0.0


def get_detailed_kernel_feedback(item: Dict) -> Dict[str, any]:
    """
    Get detailed feedback about the generated kernel code.
    
    Returns a dictionary with detailed evaluation results.
    """
    feedback = {
        "reward": 0.0,
        "has_code_blocks": False,
        "cuda_syntax_valid": False,
        "python_syntax_valid": False,
        "has_required_functions": False,
        "cuda_complete": False,
        "optimization_techniques": [],
        "errors": []
    }
    
    try:
        messages = item.get("messages", [])
        response = ""
        
        for msg in messages:
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                break
        
        if not response:
            feedback["errors"].append("No response found")
            return feedback
        
        code_blocks = extract_code_blocks(response)
        feedback["has_code_blocks"] = len(code_blocks) > 0
        
        if code_blocks:
            # Analyze the best code block
            for code in code_blocks:
                cuda_valid, cuda_errors = check_cuda_kernel_syntax(code)
                python_valid, python_errors = check_python_syntax(code)
                functions_valid, function_errors = check_required_functions(code)
                cuda_complete, completeness_errors = check_cuda_completeness(code)
                
                if cuda_valid:
                    feedback["cuda_syntax_valid"] = True
                if python_valid:
                    feedback["python_syntax_valid"] = True
                if functions_valid:
                    feedback["has_required_functions"] = True
                if cuda_complete:
                    feedback["cuda_complete"] = True
                
                # Check optimization techniques
                optimizations = []
                if "__shared__" in code:
                    optimizations.append("shared_memory")
                if "__syncthreads()" in code:
                    optimizations.append("thread_sync")
                if "#pragma unroll" in code:
                    optimizations.append("loop_unrolling")
                
                feedback["optimization_techniques"] = optimizations
                feedback["errors"].extend(cuda_errors + python_errors + function_errors + completeness_errors)
        
        feedback["reward"] = get_kernel_code_reward(item)
        
    except Exception as e:
        feedback["errors"].append(f"Evaluation error: {e}")
    
    return feedback