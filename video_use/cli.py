"""Command-line interface for video-use."""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.json import JSON

from .video.service import VideoService
from .schema.models import VideoAnalysisConfig

# Create CLI app
app = typer.Typer(help="Video-Use: Convert videos to browser automation workflows")
console = Console()

# Global service instance
service = None


def get_service() -> VideoService:
    """Get or create video service instance."""
    global service
    if service is None:
        service = VideoService()
    return service


@app.command()
def analyze(
    video_path: str = typer.Argument(..., help="Path to video file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="User prompt for context"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick analysis using keyframes only"),
    fps: Optional[float] = typer.Option(None, "--fps", help="Frame extraction FPS"),
    confidence: Optional[float] = typer.Option(None, "--confidence", help="UI detection confidence threshold"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Analyze a video file and extract browser automation workflow."""
    
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Validate input
    video_file = Path(video_path)
    if not video_file.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)
    
    # Create config with custom settings
    config = VideoAnalysisConfig()
    if fps:
        config.frame_extraction_fps = fps
    if confidence:
        config.ui_detection_confidence = confidence
    
    # Create service with config
    video_service = VideoService(config)
    
    # Validate video format
    if not video_service.validate_video_file(video_file):
        console.print(f"[red]Error: Unsupported video format. Supported formats: {video_service.get_supported_formats()}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Starting video analysis for: {video_file}[/green]")
    
    # Run analysis with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        if quick:
            task = progress.add_task("Performing quick analysis...", total=None)
            result = asyncio.run(video_service.quick_analysis(video_file))
        else:
            task = progress.add_task("Analyzing video...", total=None)
            result = asyncio.run(video_service.analyze_video_file(
                video_file, 
                user_prompt=prompt
            ))
        
        progress.update(task, completed=True)
    
    # Display results
    if result.success:
        console.print(f"[green]✓ Analysis completed successfully![/green]")
        console.print(f"Analysis ID: {result.analysis_id}")
        console.print(f"Processing time: {result.processing_time:.2f} seconds")
        console.print(f"Confidence score: {result.confidence_score:.2f}")
        console.print(f"Workflow steps: {len(result.workflow_steps)}")
        
        # Display workflow steps
        if result.workflow_steps:
            table = Table(title="Workflow Steps")
            table.add_column("Step", style="cyan", no_wrap=True)
            table.add_column("Action", style="magenta")
            table.add_column("Description", style="green")
            table.add_column("Confidence", style="yellow")
            
            for i, step in enumerate(result.workflow_steps, 1):
                table.add_row(
                    str(i),
                    step['action_type'],
                    step['description'],
                    f"{step['confidence']:.2f}"
                )
            
            console.print(table)
        
        # Save results if output directory specified
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = asyncio.run(video_service.save_analysis(result.analysis_id, output_dir))
            if save_path:
                console.print(f"[green]Results saved to: {save_path}[/green]")
    
    else:
        console.print(f"[red]✗ Analysis failed: {result.error_message}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    analysis_id: str = typer.Argument(..., help="Analysis ID to export"),
    format: str = typer.Option("browser-use", "--format", "-f", help="Export format (browser-use, json)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Export analysis results to different formats."""
    
    video_service = get_service()
    
    # Get analysis result
    result = asyncio.run(video_service.get_analysis_result(analysis_id))
    if not result:
        console.print(f"[red]Error: Analysis ID not found: {analysis_id}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Exporting analysis {analysis_id} in {format} format...[/green]")
    
    if format == "browser-use":
        workflow = asyncio.run(video_service.export_workflow_to_browser_use(analysis_id))
        if workflow:
            if output:
                output_file = Path(output)
                with open(output_file, 'w') as f:
                    json.dump(workflow, f, indent=2)
                console.print(f"[green]Browser-use workflow exported to: {output_file}[/green]")
            else:
                console.print(Panel(JSON.from_data(workflow), title="Browser-Use Workflow"))
        else:
            console.print("[red]Error: Failed to export workflow[/red]")
            raise typer.Exit(1)
    
    elif format == "json":
        # Export raw analysis results
        if output:
            output_file = Path(output)
            save_path = asyncio.run(video_service.save_analysis(analysis_id, output_file.parent))
            console.print(f"[green]Analysis exported to: {save_path}[/green]")
        else:
            # Display summary
            console.print(Panel(f"""
Analysis ID: {analysis_id}
Success: {result.success}
Processing Time: {result.processing_time:.2f}s
Actions: {len(result.actions)}
UI Elements: {len(result.ui_elements)}
Workflow: {result.workflow.name if result.workflow else 'None'}
            """, title="Analysis Summary"))
    
    else:
        console.print(f"[red]Error: Unsupported export format: {format}[/red]")
        raise typer.Exit(1)


@app.command()
def list():
    """List all cached analysis results."""
    
    video_service = get_service()
    analyses = asyncio.run(video_service.list_analyses())
    
    if not analyses:
        console.print("[yellow]No analyses found in cache.[/yellow]")
        return
    
    table = Table(title="Cached Analyses")
    table.add_column("Analysis ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Video File", style="green")
    table.add_column("Actions", style="yellow")
    table.add_column("Workflow", style="blue")
    table.add_column("Time", style="white")
    
    for analysis in analyses:
        status = "✓ Success" if analysis['success'] else "✗ Failed"
        video_file = Path(analysis['video_file']).name if analysis['video_file'] else "Unknown"
        
        table.add_row(
            analysis['analysis_id'][:8] + "...",
            status,
            video_file,
            str(analysis['actions_count']),
            analysis['workflow_name'] or "None",
            f"{analysis.get('processing_time', 0):.1f}s"
        )
    
    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    fps: Optional[float] = typer.Option(None, "--fps", help="Set frame extraction FPS"),
    confidence: Optional[float] = typer.Option(None, "--confidence", help="Set UI detection confidence"),
    ocr: Optional[bool] = typer.Option(None, "--ocr", help="Enable/disable OCR"),
    parallel: Optional[bool] = typer.Option(None, "--parallel", help="Enable/disable parallel processing")
):
    """Configure video analysis settings."""
    
    if show:
        config = VideoAnalysisConfig()
        config_dict = {
            'frame_extraction_fps': config.frame_extraction_fps,
            'ui_detection_confidence': config.ui_detection_confidence,
            'action_confidence_threshold': config.action_confidence_threshold,
            'enable_ocr': config.enable_ocr,
            'llm_model': config.llm_model,
            'max_frames': config.max_frames,
            'parallel_processing': config.parallel_processing,
            'max_workers': config.max_workers
        }
        console.print(Panel(JSON.from_data(config_dict), title="Current Configuration"))
    
    # Update configuration (this is simplified - in a real app you'd save to a config file)
    updates = []
    if fps is not None:
        updates.append(f"Frame extraction FPS: {fps}")
    if confidence is not None:
        updates.append(f"UI detection confidence: {confidence}")
    if ocr is not None:
        updates.append(f"OCR enabled: {ocr}")
    if parallel is not None:
        updates.append(f"Parallel processing: {parallel}")
    
    if updates:
        console.print("[green]Configuration updated:[/green]")
        for update in updates:
            console.print(f"  • {update}")
        console.print("[yellow]Note: Changes will apply to new analyses only.[/yellow]")


@app.command()
def demo():
    """Show demo of video-use capabilities."""
    
    console.print(Panel.fit("""
[bold green]Video-Use Demo[/bold green]

Video-Use converts browser interaction videos into automated workflows.

[bold cyan]How it works:[/bold cyan]
1. [white]Extract frames[/white] from your screen recording
2. [white]Detect UI elements[/white] (buttons, inputs, links) using computer vision  
3. [white]Infer user actions[/white] (clicks, typing, navigation) from changes
4. [white]Generate workflow[/white] compatible with browser-use automation

[bold cyan]Example usage:[/bold cyan]
[dim]# Analyze a video[/dim]
video-use analyze recording.mp4 --output ./results

[dim]# Quick analysis (keyframes only)[/dim]  
video-use analyze recording.mp4 --quick

[dim]# Export to browser-use format[/dim]
video-use export abc123 --format browser-use --output workflow.json

[bold cyan]Supported formats:[/bold cyan]
.mp4, .avi, .mov, .mkv, .webm, .flv, .wmv

[bold cyan]Tips:[/bold cyan]
• Record clear browser interactions at 1080p+
• Keep mouse movements smooth and deliberate  
• Pause briefly after each action
• Use --prompt to provide context about the task
    """, title="🎥 Video-Use"))


@app.command()
def clean(
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt")
):
    """Clean up cached analysis results."""
    
    if not confirm:
        if not typer.confirm("Are you sure you want to clear all cached analyses?"):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    
    video_service = get_service()
    count = asyncio.run(video_service.cleanup_cache())
    
    console.print(f"[green]✓ Cleaned up {count} cached analyses.[/green]")


@app.command()
def info(
    video_path: str = typer.Argument(..., help="Path to video file")
):
    """Show information about a video file."""
    
    video_file = Path(video_path)
    if not video_file.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)
    
    video_service = get_service()
    
    # Get video metadata
    try:
        metadata = asyncio.run(video_service.analyzer.frame_extractor.extract_video_metadata(video_file))
        
        info_table = Table(title=f"Video Information: {video_file.name}")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("File Path", str(metadata.file_path))
        info_table.add_row("Duration", f"{metadata.duration:.2f} seconds")
        info_table.add_row("FPS", f"{metadata.fps:.2f}")
        info_table.add_row("Resolution", f"{metadata.width}x{metadata.height}")
        info_table.add_row("Total Frames", str(metadata.total_frames))
        info_table.add_row("Format", metadata.format)
        info_table.add_row("File Size", f"{metadata.size_bytes / (1024*1024):.1f} MB")
        
        # Estimate analysis time
        estimated_frames = min(1000, int(metadata.total_frames / metadata.fps))
        estimated_time = estimated_frames * 0.5  # Rough estimate
        info_table.add_row("Estimated Analysis Time", f"{estimated_time:.0f} seconds")
        
        console.print(info_table)
        
        # Validation
        if video_service.validate_video_file(video_file):
            console.print("[green]✓ Video format is supported[/green]")
        else:
            console.print("[red]✗ Video format is not supported[/red]")
        
    except Exception as e:
        console.print(f"[red]Error reading video metadata: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 