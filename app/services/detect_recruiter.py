import re
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_recruiter(input_data: Dict[str, any]) -> Dict[str, any]:
    
    user_message = input_data["input"].lower()
    metadata = input_data.get("metadata", {})
    state = input_data.get("state", {"is_recruiter": False})
    
    logger.info(f"Processing input for recruiter detection: {user_message}")
    logger.info(f"Metadata: {metadata}")
    
    # Recruiter-related patterns
    recruiter_patterns = [
        r"\b(recruiter|hr|human\s*resources|talent\s*acquisition|hiring\s*manager)\b",
        r"\b(job|position|role|opportunity|vacancy|career|employment|interview)\b",
        r"\b(resume|cv|portfolio|experience|qualifications|skills|certifications)\b",
        r"\b(available|availability|start\s*date|salary|compensation)\b",
        r"\b(company|employer|organization|team)\b\s*(is\s*looking|seeks|needs)\b",
        r"\b(technical\s*skills|professional\s*background|work\s*experience)\b",
        r"\b(internship|full\s*time|part\s*time|contract|freelance)\b",
    ]
    
    # Patterns for recruiter-like questions
    question_patterns = [
        r"\b(what|tell\s*me)\s*(is|are|about)\s*(your\s*)?(skills|experience|projects|background|education|certifications)\b",
        r"\b(have\s*you|did\s*you)\s*(worked|built|developed|contributed)\b",
        r"\b(why|how)\s*(did\s*you|do\s*you)\s*(choose|approach|learn)\b",
        r"\b(are\s*you)\s*(available|looking\s*for|interested\s*in)\s*(a\s*)?(job|role|position)\b",
        r"\b(what\s*is|tell\s*me)\s*(your\s*)?(strengths|weaknesses|achievements|goals)\b",
    ]
    
    # Check for recruiter patterns in user message
    is_recruiter_detected = any(re.search(pattern, user_message) for pattern in recruiter_patterns)
    is_recruiter_question = any(re.search(pattern, user_message) for pattern in question_patterns)
    
    # Analyze metadata (e.g., user agent, session ID, or email domain if available)
    metadata_hints = False
    if metadata:
        user_agent = metadata.get("user_agent", "").lower()
        email = metadata.get("email", "").lower()
        if user_agent and any(keyword in user_agent for keyword in ["linkedin", "recruit", "hr"]):
            metadata_hints = True
        if email and any(domain in email for domain in ["@linkedin.com", "@recruitment", "@hr", "@talent"]):
            metadata_hints = True
    
    # Update recruiter state
    if is_recruiter_detected or is_recruiter_question or metadata_hints:
        state["is_recruiter"] = True
        logger.info("Recruiter detected based on patterns or metadata")
    else:
        state["is_recruiter"] = False
        logger.info("No recruiter detected, using casual tone")
    
    logger.info(f"Updated state: {state}")
    
    return {
        "input": input_data["input"],
        "profile": input_data.get("profile", {}),
        "state": state,
        "metadata": metadata,
        "message_history": input_data.get("message_history", [])
    }