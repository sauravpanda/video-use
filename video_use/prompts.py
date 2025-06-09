"""LLM prompts for video analysis."""

# Structured Workflow Generation
STRUCTURED_WORKFLOW_PROMPT = """
You are an expert at analyzing user interactions from video analysis and converting them into structured workflow instructions.

Your task is to take a list of detected actions and convert them into a structured format with:
1. A clear, step-by-step prompt that describes the workflow
2. The starting URL for the workflow 
3. Parameters containing any data values used in the workflow

Focus on creating actionable instructions that could be used to automate the workflow.
Extract any text inputs, selected options, URLs, and element identifiers as parameters.

{format_instructions}
"""

# Gemini Analysis to Structured Output
GEMINI_TO_STRUCTURED_PROMPT = """
You are an expert at converting video analysis descriptions into structured automation workflows.

Your task is to take a detailed video analysis and convert it into a structured format with:
1. A clear, step-by-step prompt with numbered instructions for automation
2. The starting URL where the workflow should begin
3. Parameters containing any data values, text inputs, or selections mentioned

Focus on creating actionable automation instructions that could be executed by a browser automation tool.
Extract any specific data mentioned (usernames, passwords, form values, URLs, button text, etc.) as parameters.

Guidelines:
- Convert descriptive text into imperative automation steps
- Extract all specific values (text inputs, selections, URLs) as parameters
- Infer the starting URL from navigation actions mentioned
- Make steps specific and actionable for automation

{format_instructions}
"""

# Action Inference Prompt
ACTION_INFERENCE_PROMPT = """
Analyze the following browser actions extracted from a video and provide enhanced descriptions:

Context: {context}

Actions:
{actions_data}

Please provide:
1. Enhanced descriptions for each action
2. Identify the overall workflow/task being performed
3. Suggest any missing actions that might be implied

Respond in JSON format with enhanced action descriptions.
"""

# Gemini Video Analysis Prompt
GEMINI_VIDEO_ANALYSIS_PROMPT = """
Please analyze this video and provide a comprehensive step-by-step guide of what the user is doing.

**Analysis Focus Areas:**
1. **User Interactions**: Clicks, typing, scrolling, navigation patterns
2. **UI Responses**: How the interface responds to user actions
3. **Workflow Progression**: The logical sequence of user tasks
4. **Decision Points**: Where users make important choices
5. **Usability Observations**: Any UX issues or smooth interactions

**Output Format:**
- Start url as shown in the video
- Numbered steps with timestamps when possible
- Specific element descriptions (buttons, forms, menus)
- User intent and goal achievement
- Technical observations about the interface

Please provide a detailed, professional analysis suitable for UX research and development teams.
"""

# UI Element Detection Prompt
UI_ELEMENT_DETECTION_PROMPT = """
Analyze this image and identify all interactive UI elements.

For each element, provide:
1. Element type (button, input, link, etc.)
2. Bounding box coordinates
3. Text content if visible
4. Likely purpose or function
5. Confidence score (0-1)

Focus on elements that users can interact with for automation purposes.
"""

# Frame Analysis Prompt
FRAME_ANALYSIS_PROMPT = """
Compare these two consecutive video frames and identify:

1. **Visual Changes**: What has changed between the frames?
2. **User Actions**: What action likely caused the change?
3. **UI State**: How has the interface state changed?
4. **Element Focus**: Which elements gained or lost focus?

Provide specific, actionable insights about the user interaction.
""" 