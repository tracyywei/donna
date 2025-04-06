import asyncio
import random
import sounddevice as sd

import numpy as np

from agents import Agent, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoicePipeline,
)

from util import AudioPlayer, record_audio

"""
1. You can record an audio clip in the terminal.
2. The pipeline automatically transcribes the audio.
3. The agent workflow is a simple one that starts at the Assistant agent.
4. The output of the agent is streamed to the audio player.
"""

@function_tool
def parse_response(user_text: str) -> str:
    return user_text

@function_tool
def follow_up(classification: str) -> str:
    return classification

@function_tool
def produce_output(classification: str) -> str:
    return classification

@function_tool
def produce_intake(classification: str) -> str:
    return classification

@function_tool
def produce_resources(classification: str) -> str:
    return classification

with open("initial_instructions.txt", "r") as file:
    initial_instructions = file.read()

agent = Agent(
    name="Classify-Problem Agent",
    instructions=prompt_with_handoff_instructions(
        initial_instructions,
    ),
    model="gpt-4o",
)


class WorkflowCallbacks(SingleAgentWorkflowCallbacks):
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        print(f"[debug] on_run called with transcription: {transcription}")


async def main():
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent, callbacks=WorkflowCallbacks())
    )

    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)
    print(result)

    with AudioPlayer() as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                sd.play(event.data, samplerate=24000)
                sd.wait()
                player.add_audio(event.data)
                print("Received audio")
            elif event.type == "voice_stream_event_lifecycle":
                print(f"Received lifecycle event: {event.event}")

        # Add 1 second of silence to the end of the stream to avoid cutting off the last audio.
        player.add_audio(np.zeros(24000 * 1, dtype=np.int16))


if __name__ == "__main__":
    asyncio.run(main())
