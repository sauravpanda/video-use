"""Workflow execution service using browser-use."""

import asyncio
import logging
import os
import uuid
from typing import Optional, Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from browser_use.browser import BrowserProfile

from ..models.workflow import StructuredWorkflowOutput
from ..models.api import WorkflowExecutionResponse

logger = logging.getLogger(__name__)


class WorkflowExecutionService:
    """Service for executing workflows using browser-use agent."""
    
    def __init__(self):
        self.active_executions: Dict[str, Any] = {}
        logger.info("WorkflowExecutionService initialized")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    
    async def execute_workflow(
        self,
        workflow: StructuredWorkflowOutput,
        execution_id: Optional[str] = None,
        headless: bool = False,
        timeout: int = 30
    ) -> WorkflowExecutionResponse:
        """
        Execute a structured workflow using browser-use agent.
        
        Args:
            workflow: The workflow to execute
            execution_id: Optional execution ID for tracking
            headless: Whether to run browser in headless mode
            timeout: Timeout in seconds
            
        Returns:
            WorkflowExecutionResponse with execution results
        """
        if not execution_id:
            execution_id = str(uuid.uuid4())
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Starting workflow execution {execution_id}")
            
            self.browser_profile = BrowserProfile(
                window_size={"width": 1920, "height": 1080},
                headless=headless,
            )
            # Initialize browser-use agent
            agent = Agent(
                task=workflow.prompt,
                llm=self.llm,
                browser_profile=self.browser_profile,
            )
            
            # Store execution info
            self.active_executions[execution_id] = {
                "workflow": workflow,
                "start_time": start_time,
                "status": "running"
            }
            
            # Execute the workflow
            logger.info(f"Executing workflow: {workflow.prompt}")
            
            # Create task with timeout
            execution_task = asyncio.create_task(
                agent.run()
            )
            
            try:
                results = await asyncio.wait_for(execution_task, timeout=timeout)
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Update execution status
                self.active_executions[execution_id]["status"] = "completed"
                
                logger.info(f"Workflow execution {execution_id} completed successfully")
                
                return WorkflowExecutionResponse(
                    success=True,
                    execution_id=execution_id,
                    results=[{"workflow_result": str(results)}],
                    execution_time=execution_time
                )
                
            except asyncio.TimeoutError:
                execution_task.cancel()
                raise Exception(f"Workflow execution timed out after {timeout} seconds")
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Workflow execution {execution_id} failed: {e}")
            
            # Update execution status
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = "failed"
            
            return WorkflowExecutionResponse(
                success=False,
                execution_id=execution_id,
                results=[],
                execution_time=execution_time,
                error_message=str(e)
            )
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution."""
        return self.active_executions.get(execution_id)
    
    def list_active_executions(self) -> List[str]:
        """List all active execution IDs."""
        return list(self.active_executions.keys()) 