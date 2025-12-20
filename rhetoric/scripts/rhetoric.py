#!/usr/bin/env python3
"""
Rhetoric: The Reasoning Engine
CLI entrypoint for structured reasoning and deliberation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

# Add parent directory to path so we can import local modules
# Structure: rhetoric/scripts/rhetoric.py, rhetoric/ai-counsel/, rhetoric/sequentialthinking_py/, etc.
_rhetoric_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_rhetoric_dir))
sys.path.insert(0, str(_rhetoric_dir / "ai-counsel"))

from ai_counsel.client import AICounselClient, Participant
from ai_counsel.types import DeliberationResult
from sequentialthinking_py.client import SequentialThinkingClient
from sequentialthinking_py.types import ThoughtInput, ThoughtData
from vibecheck_py.client import VibeCheckClient, VibeCheckInput, VibeCheckResponse

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger("rhetoric")

THOUGHTS_FILE = Path("thoughts.json")

@dataclass
class ThoughtRecord:
    id: str
    timestamp: str
    session_id: str
    content: str
    type: str
    revision_of: Optional[str] = None
    thought_number: int = 0
    total_thoughts: int = 0

class ThinkingManager:
    """Manages persistence for thoughts."""
    
    def __init__(self, filepath: Path = THOUGHTS_FILE):
        self.filepath = filepath
        self.history: List[dict] = []
        self.load()

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    self.history = json.load(f)
            except json.JSONDecodeError:
                self.history = []
        else:
            self.history = []

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_thought(self, thought_data: ThoughtData, session_id: str):
        import uuid
        from datetime import datetime
        
        thought_id = f"th-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        record = {
            "id": thought_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "content": thought_data.thought,
            "type": "analysis", # Default type
            "revision_of": None, # Needs logic to link if it is a revision
            "thought_number": thought_data.thought_number,
            "total_thoughts": thought_data.total_thoughts,
            "is_revision": thought_data.is_revision,
            "revises_thought": thought_data.revises_thought,
            "branch_from_thought": thought_data.branch_from_thought,
            "branch_id": thought_data.branch_id
        }
        
        if thought_data.is_revision and thought_data.revises_thought:
            # Find the ID of the thought being revised (simplified logic: find last thought with that number in session)
            # ideally client provides the ID, but sequentialthinking_py uses numbers.
            pass

        self.history.append(record)
        self.save()
        return thought_id

    def get_session_thoughts(self, session_id: str) -> List[dict]:
        return [t for t in self.history if t.get("session_id") == session_id]

    def get_stats(self):
        sessions = set(t.get("session_id") for t in self.history)
        return {
            "total_thoughts": len(self.history),
            "active_sessions": len(sessions)
        }

async def cmd_think(args):
    """Execute think command."""
    client = SequentialThinkingClient(disable_logging=True)
    manager = ThinkingManager()
    
    # Determine thought number if not provided
    session_thoughts = manager.get_session_thoughts(args.session_id)
    current_thought_number = len(session_thoughts) + 1
    
    if args.revision_of:
        # If revising, we need to find the thought number of the revised thought
        # This is a simplification. Real implementation would look up ID.
        pass

    input_data = ThoughtInput(
        thought=args.thought,
        thought_number=current_thought_number,
        total_thoughts=max(current_thought_number, 5), # Default total
        next_thought_needed=True,
        is_revision=bool(args.revision_of),
        branch_from_thought=None # Needs parsing from args if supported
    )

    result = client.process_thought(input_data)
    
    if result.is_ok():
        resp = result.value
        # Use the data from the input mostly, as response is minimal
        # Create a ThoughtData object to pass to manager
        t_data = ThoughtData(
            thought=args.thought,
            thought_number=resp.thought_number,
            total_thoughts=resp.total_thoughts,
            next_thought_needed=resp.next_thought_needed,
            is_revision=input_data.is_revision,
            revises_thought=input_data.revises_thought,
            branch_from_thought=input_data.branch_from_thought,
            branch_id=input_data.branch_id,
            needs_more_thoughts=input_data.needs_more_thoughts
        )
        
        thought_id = manager.add_thought(t_data, args.session_id)

        print(json.dumps({
            "status": "success",
            "thought_id": thought_id,
            "session_id": args.session_id,
            "thought_number": resp.thought_number,
            "total_thoughts": resp.total_thoughts,
            "thought": args.thought,  # Include the thought content for downstream processing
            "next_thought_needed": resp.next_thought_needed
        }, indent=2))
    else:
        print(json.dumps({
            "status": "error",
            "message": result.error.message
        }, indent=2))
        sys.exit(1)

async def cmd_deliberate(args):
    """Execute deliberate command."""
    # Check for API keys and build available models list
    available_models = []
    if os.environ.get("OPENAI_API_KEY"):
        available_models.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available_models.append("anthropic")
    if os.environ.get("OPENROUTER_API_KEY"):
        available_models.append("openrouter")
    if os.environ.get("OLLAMA_URL"):
        available_models.append("ollama")

    if len(available_models) < 2 and not args.allow_single:
        print(json.dumps({
            "status": "error",
            "code": 4,
            "message": "Insufficient models for deliberation. Configure at least 2 of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OLLAMA_URL"
        }, indent=2))
        sys.exit(1)

    # Setup participants based on available keys
    participants = []
    if "openai" in available_models:
        participants.append(Participant(adapter="openai", model="gpt-4o"))
    if "anthropic" in available_models:
        participants.append(Participant(adapter="anthropic", model="claude-sonnet-4-20250514"))
    if "openrouter" in available_models and len(participants) < 2:
        participants.append(Participant(adapter="openrouter", model="google/gemini-flash-1.5"))
    if "ollama" in available_models and len(participants) < 2:
        participants.append(Participant(adapter="ollama", model="llama3.2"))

    if len(participants) < 2:
        print(json.dumps({
            "status": "error",
            "code": 4,
            "message": "Could not configure 2 participants for deliberation"
        }, indent=2))
        sys.exit(1)

    client = AICounselClient(enable_transcripts=True, enable_decision_graph=False)

    try:
        result = await client.deliberate(
            question=args.question,
            participants=participants[:2],  # Use first 2 participants
            rounds=args.rounds,
            context=args.context
        )

        if result.is_ok():
            val = result.value
            # Calculate confidence from convergence info if available
            confidence = 0.85
            if val.convergence_info and val.convergence_info.final_similarity:
                confidence = val.convergence_info.final_similarity

            print(json.dumps({
                "status": "completed",
                "question": args.question,
                "rounds_completed": val.rounds_completed,
                "consensus": val.summary.consensus,
                "confidence": round(confidence, 2)
            }, indent=2))
        else:
            print(json.dumps({
                "status": "error",
                "message": result.error.message
            }, indent=2))
            sys.exit(1)
    finally:
        await client.close()

async def cmd_review(args):
    """Execute review command."""
    manager = ThinkingManager()
    thoughts = manager.get_session_thoughts(args.session_id)

    # Analyze patterns
    revisions = sum(1 for t in thoughts if t.get("is_revision"))
    branches = sum(1 for t in thoughts if t.get("branch_from_thought"))
    count = len(thoughts)

    revision_rate = revisions / count if count > 0 else 0

    detected_issues = []
    recommendations = []

    # Use VibeCheck for pattern analysis if we have enough thoughts
    if count >= 3:
        try:
            # Build summary of thinking for vibe check
            thought_summary = "\n".join(
                f"{i+1}. {t.get('content', '')[:100]}"
                for i, t in enumerate(thoughts[-5:])  # Last 5 thoughts
            )

            client = VibeCheckClient()
            vibe_input = VibeCheckInput(
                goal="Review thinking patterns for cognitive biases",
                plan=f"Analyzed {count} thoughts with {revision_rate:.0%} revision rate",
                progress=thought_summary,
                session_id=args.session_id
            )

            result = await client.vibe_check(vibe_input)

            if result.is_ok():
                # Parse vibe check response for issues
                response_text = result.value.questions.lower()

                # Detect common patterns from vibe check output
                if "analysis paralysis" in response_text or revision_rate > 0.4:
                    detected_issues.append("analysis_paralysis")
                if "scope" in response_text or "creep" in response_text:
                    detected_issues.append("scope_creep")
                if "assumption" in response_text:
                    detected_issues.append("unchecked_assumptions")
                if "bias" in response_text:
                    detected_issues.append("confirmation_bias")

        except Exception:
            pass  # Fall back to basic analysis

    # Generate recommendations based on patterns
    if revision_rate > 0.3:
        recommendations.append("High revision rate suggests indecision - consider narrowing scope")
    if branches > 2:
        recommendations.append("Multiple branches detected - consider converging on main path")
    if count < 3:
        recommendations.append("Keep thinking - more analysis needed")
    elif count > 10 and not detected_issues:
        recommendations.append("Consider converging toward a decision")
    if "analysis_paralysis" in detected_issues:
        recommendations.append("Analysis paralysis detected - make a decision to proceed")

    if not recommendations:
        recommendations.append("Thinking patterns look healthy")

    print(json.dumps({
        "status": "success",
        "session_id": args.session_id,
        "thought_count": count,
        "patterns": {
            "revision_rate": round(revision_rate, 2),
            "branch_count": branches,
            "detected_issues": detected_issues
        },
        "recommendations": recommendations
    }, indent=2))

async def cmd_status(args):
    """Execute status command."""
    manager = ThinkingManager()
    stats = manager.get_stats()

    # Count available models with providers info
    available_providers = []
    if os.environ.get("OPENAI_API_KEY"):
        available_providers.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")
    if os.environ.get("OPENROUTER_API_KEY"):
        available_providers.append("openrouter")
    if os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        available_providers.append("gemini")
    if os.environ.get("OLLAMA_URL"):
        available_providers.append("ollama")

    models = len(available_providers)

    print(json.dumps({
        "active_sessions": stats["active_sessions"],
        "total_thoughts": stats["total_thoughts"],
        "models_available": models,
        "providers": available_providers,
        "deliberation_ready": models >= 2,
        "last_activity": manager.history[-1]["timestamp"] if manager.history else None
    }, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Rhetoric: The Reasoning Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Think
    think_parser = subparsers.add_parser("think", help="Record a thought")
    think_parser.add_argument("thought", help="The thought content")
    think_parser.add_argument("--session-id", default="default", help="Session ID")
    think_parser.add_argument("--revision-of", help="ID of thought to revise")
    think_parser.add_argument("--branch-from", help="ID of thought to branch from")

    # Deliberate
    deliberate_parser = subparsers.add_parser("deliberate", help="Deliberate on a question")
    deliberate_parser.add_argument("question", help="Question to deliberate")
    deliberate_parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    deliberate_parser.add_argument("--context", help="Context for deliberation")
    deliberate_parser.add_argument("--debug", action="store_true", help="Show debug info")
    deliberate_parser.add_argument("--allow-single", action="store_true", help="Allow single model (dev)")

    # Review
    review_parser = subparsers.add_parser("review", help="Review session")
    review_parser.add_argument("--session-id", default="default", help="Session ID")

    # Status
    status_parser = subparsers.add_parser("status", help="Get system status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check configuration before running
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from shared.config_check import require_skill_config
        require_skill_config("rhetoric")
    except ImportError:
        pass  # shared module not available, skip check

    if args.command == "think":
        asyncio.run(cmd_think(args))
    elif args.command == "deliberate":
        asyncio.run(cmd_deliberate(args))
    elif args.command == "review":
        asyncio.run(cmd_review(args))
    elif args.command == "status":
        asyncio.run(cmd_status(args))

if __name__ == "__main__":
    main()
