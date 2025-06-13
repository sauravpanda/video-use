"""
CSV Batch Execution Example for Video-Use.
Analyzes video once, then executes the workflow multiple times with data from CSV.

Usage:
    python csv_batch_execution_example.py <video_path> [csv_path]
    python csv_batch_execution_example.py sample_form_filling.mp4
    python csv_batch_execution_example.py login_demo.mp4 user_data.csv
"""

import argparse
import asyncio
import csv
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from video_use import VideoUseService, VideoAnalysisConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVBatchProcessor:
    """Handles batch processing of workflows with CSV data."""
    
    def __init__(self, service: VideoUseService):
        self.service = service
        self.workflow_template = None
        
    async def analyze_video_for_template(
        self, 
        video_path: Path, 
        template_start_url: str = None,
        user_provided_url: str = None
    ) -> bool:
        """Analyze video to create workflow template."""
        try:
            logger.info(f"Analyzing video for workflow template: {video_path}")
            
            # Analyze video with Gemini
            analysis_result = await self.service.analyze_video_file(
                video_path,
                use_gemini=True
            )
            
            if not analysis_result.success:
                logger.error(f"Video analysis failed: {analysis_result.error_message}")
                return False
            
            # Generate structured workflow template - let Gemini extract URL first
            analysis_text = analysis_result.workflow_steps[0].get('analysis_text', '')
            temp_workflow = await self.service.generate_structured_workflow_from_gemini(
                analysis_text,
                start_url=template_start_url or "https://example.com"
            )
            
            # Priority: User provided URL > Gemini extracted URL > Default URL
            final_url = None
            if user_provided_url and user_provided_url != "https://example.com/login":
                final_url = user_provided_url
                logger.info(f"🎯 Using USER-PROVIDED URL: {final_url}")
            elif temp_workflow.start_url and temp_workflow.start_url != "https://example.com":
                final_url = temp_workflow.start_url
                logger.info(f"🤖 Using GEMINI-EXTRACTED URL: {final_url}")
            else:
                final_url = user_provided_url or "https://example.com"
                logger.info(f"🔗 Using DEFAULT URL: {final_url}")
            
            # Create final workflow template with prioritized URL
            self.workflow_template = await self.service.generate_structured_workflow_from_gemini(
                analysis_text,
                start_url=final_url
            )
            
            logger.info("✅ Workflow template created successfully")
            logger.info(f"📝 Template prompt: {self.workflow_template.prompt[:100]}...")
            logger.info(f"🌐 Final start URL: {self.workflow_template.start_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workflow template: {e}")
            return False
    
    def load_csv_data(self, csv_path: Path) -> List[Dict[str, str]]:
        """Load data from CSV file."""
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                data_rows = list(reader)
                
            logger.info(f"✅ Loaded {len(data_rows)} rows from CSV: {csv_path}")
            
            # Log first few column names for reference
            if data_rows:
                columns = list(data_rows[0].keys())
                logger.info(f"CSV columns: {columns}")
            
            return data_rows
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            return []
    
    def customize_workflow_for_row(
        self, 
        row_data: Dict[str, str], 
        start_url_template: str = None
    ) -> 'StructuredWorkflowOutput':
        """Create customized workflow for a specific data row."""
        from video_use.models import StructuredWorkflowOutput, TokenUsage
        
        if not self.workflow_template:
            raise ValueError("No workflow template available. Run analyze_video_for_template first.")
        
        # Start with template
        customized_prompt = self.workflow_template.prompt
        customized_start_url = start_url_template or self.workflow_template.start_url
        customized_parameters = self.workflow_template.parameters.copy()
        
        # Replace placeholders in prompt with actual data
        for key, value in row_data.items():
            placeholder = f"{{{key}}}"  # e.g., {username}, {email}, {password}
            customized_prompt = customized_prompt.replace(placeholder, value)
            customized_start_url = customized_start_url.replace(placeholder, value)
            
            # Add to parameters
            customized_parameters[key] = value
        
        # Also try common variations
        for key, value in row_data.items():
            # Try uppercase placeholders
            placeholder_upper = f"{{{key.upper()}}}"
            customized_prompt = customized_prompt.replace(placeholder_upper, value)
            customized_start_url = customized_start_url.replace(placeholder_upper, value)
            
            # Try with underscores replaced with spaces
            key_spaced = key.replace('_', ' ')
            placeholder_spaced = f"{{{key_spaced}}}"
            customized_prompt = customized_prompt.replace(placeholder_spaced, value)
        
        return StructuredWorkflowOutput(
            prompt=customized_prompt + f"All variables for missing or incorrect values {row_data}",
            start_url=customized_start_url,
            parameters=customized_parameters,
            token_usage=TokenUsage()
        )
    
    async def execute_batch(
        self, 
        csv_data: List[Dict[str, str]], 
        start_url_template: str = None,
        headless: bool = True,
        timeout: int = 60,
        max_concurrent: int = 4,
        use_shared_session: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute workflow for each row in CSV data."""
        results = []
        
        if not self.workflow_template:
            raise ValueError("No workflow template available. Run analyze_video_for_template first.")
        
        logger.info(f"Starting batch execution for {len(csv_data)} rows")
        logger.info(f"Max concurrent executions: {max_concurrent}")
        logger.info(f"Using shared browser session: {use_shared_session}")
        
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_row(index: int, row_data: Dict[str, str]):
            async with semaphore:
                try:
                    logger.info(f"Processing row {index + 1}/{len(csv_data)}")
                    
                    # Add delay to respect API rate limits only when not using shared session
                    if not use_shared_session and index > 0:  # No delay for first request
                        await asyncio.sleep(2)  # 2 second delay between requests
                    
                    # Customize workflow for this row
                    customized_workflow = self.customize_workflow_for_row(
                        row_data, start_url_template
                    )
                    
                    # Execute workflow with shared session
                    execution_result = await self.service.execute_workflow(
                        customized_workflow,
                        headless=headless,
                        timeout=timeout,
                        use_shared_session=use_shared_session
                    )
                    
                    result = {
                        'row_index': index,
                        'row_data': row_data,
                        'execution_result': execution_result,
                        'success': execution_result.success,
                        'execution_time': execution_result.execution_time,
                        'error': execution_result.error_message if not execution_result.success else None
                    }
                    
                    if execution_result.success:
                        logger.info(f"✅ Row {index + 1} completed successfully in {execution_result.execution_time:.2f}s")
                    else:
                        logger.error(f"❌ Row {index + 1} failed: {execution_result.error_message}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Row {index + 1} failed with exception: {e}")
                    return {
                        'row_index': index,
                        'row_data': row_data,
                        'execution_result': None,
                        'success': False,
                        'execution_time': 0.0,
                        'error': str(e)
                    }
        
        try:
            # Execute all rows concurrently (with semaphore limiting)
            tasks = [
                execute_single_row(i, row_data) 
                for i, row_data in enumerate(csv_data)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Summary
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            total_time = sum(r['execution_time'] for r in results)
            
            logger.info(f"🎯 Batch execution complete!")
            logger.info(f"   ✅ Successful: {successful}")
            logger.info(f"   ❌ Failed: {failed}")
            logger.info(f"   ⏱️ Total execution time: {total_time:.2f}s")
            
            return results
            
        finally:
            # Stop shared browser session if we were using one
            if use_shared_session:
                await self.service.execution_service.stop_browser_session()


async def main(video_path: Path, csv_path: Path, args):
    """Demonstrate CSV batch processing."""
    
    print("🎬 Video-Use: CSV Batch Processing Demo")
    print("=" * 45)
    print(f"📹 Video: {video_path}")
    print(f"📊 CSV: {csv_path}")
    print()
    
    # Check if video file exists
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        print("💡 Please provide a valid video file path")
        return
    
    # Create sample CSV if it doesn't exist
    if not csv_path.exists():
        print(f"📝 Creating sample CSV file: {csv_path}")
        create_sample_csv(csv_path)
    
    # Check for required API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY environment variable not set")
        print("💡 Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return
    
    # Initialize service and batch processor
    config = VideoAnalysisConfig(
        frame_extraction_fps=1.0,
        max_frames=20
    )
    service = VideoUseService(config)
    batch_processor = CSVBatchProcessor(service)
    
    try:
        print("\n📹 Step 1: Analyzing video for workflow template...")
        print(f"🌐 Start URL provided: {args.start_url}")
        success = await batch_processor.analyze_video_for_template(
            video_path,
            template_start_url=args.start_url,  # Can use placeholders like https://{domain}/login
            user_provided_url=args.start_url    # User-provided URL gets priority
        )
        
        if not success:
            print("❌ Failed to create workflow template")
            return
        
        print("\n📊 Step 2: Loading CSV data...")
        csv_data = batch_processor.load_csv_data(csv_path)
        
        if not csv_data:
            print("❌ No data loaded from CSV")
            return
        
        print(f"✅ Loaded {len(csv_data)} rows of data")
        
        print("\n🚀 Step 3: Executing batch workflows...")
        results = await batch_processor.execute_batch(
            csv_data,
            start_url_template=args.start_url,  # Template URL
            headless=args.headless,  # Run in headless mode for batch processing
            timeout=args.timeout,     # Timeout per execution
            max_concurrent=args.max_concurrent,  # Process workflows simultaneously
            use_shared_session=args.use_shared_session  # Use shared browser session
        )
        
        print("\n📈 Batch Results Summary:")
        print("=" * 25)
        
        for i, result in enumerate(results):
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            time_taken = result['execution_time']
            row_preview = str(result['row_data'])[:50] + "..." if len(str(result['row_data'])) > 50 else str(result['row_data'])
            
            print(f"Row {i+1}: {status} ({time_taken:.2f}s) - {row_preview}")
            
            if not result['success'] and result['error']:
                print(f"       Error: {result['error']}")
        
        # Overall statistics
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r['execution_time'] for r in results)
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total rows processed: {len(results)}")
        print(f"   Successful executions: {successful}")
        print(f"   Failed executions: {len(results) - successful}")
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   Average time per row: {total_time/len(results):.2f}s")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        logger.exception("Batch processing error")


def create_sample_csv(csv_path: Path):
    """Create a sample CSV file for demonstration."""
    sample_data = [
        {
            'username': 'user1@example.com',
            'password': 'password123',
            'first_name': 'John',
            'last_name': 'Doe',
            'company': 'Acme Corp'
        },
        {
            'username': 'user2@example.com', 
            'password': 'secret456',
            'first_name': 'Jane',
            'last_name': 'Smith',
            'company': 'Tech Solutions'
        },
        {
            'username': 'user3@example.com',
            'password': 'mypass789', 
            'first_name': 'Bob',
            'last_name': 'Johnson',
            'company': 'StartupXYZ'
        }
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if sample_data:
            fieldnames = sample_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_data)
    
    print(f"✅ Created sample CSV with {len(sample_data)} rows")
    print("💡 You can modify this CSV file to include your own data")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CSV batch execution example for video-use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csv_batch_execution_example.py sample_form_filling.mp4
  python csv_batch_execution_example.py login_demo.mp4 user_data.csv
  python csv_batch_execution_example.py form_fill.mp4 --max-concurrent 3 --timeout 45

Requirements:
  - Set OPENAI_API_KEY environment variable  
  - Video file in supported format (MP4, AVI, MOV, MKV, WebM)
  - CSV file with data columns matching workflow placeholders
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        'csv_path',
        type=str,
        nargs='?',
        default='sample_data.csv',
        help='Path to CSV file with batch data (default: sample_data.csv)'
    )
    
    parser.add_argument(
        '--start-url',
        type=str,
        default='https://example.com/login',
        help='Start URL for workflows (PRIORITY: User provided > Gemini extracted > Default). Default: https://example.com/login'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=2,
        help='Maximum concurrent workflow executions (default: 1)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout per workflow execution in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode (default: True for batch processing)'
    )
    
    parser.add_argument(
        '--use-shared-session',
        action='store_true',
        default=True,
        help='Use shared browser session for concurrent executions (default: True)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    video_path = Path(args.video_path)
    csv_path = Path(args.csv_path)
    
    # Default to headless for batch processing
    if not hasattr(args, 'headless') or args.headless is None:
        args.headless = True
    
    asyncio.run(main(video_path, csv_path, args)) 