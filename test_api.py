import os
import json
from api import GameStateRecognizer

def test_api():
    repo_dir = r"C:\Users\sabih\OneDrive\Desktop\game_state_recognition\sn-gamestate"
    video_path = r"C:\Users\sabih\OneDrive\Desktop\game_state_recognition\data\SoccerNetGS\gamestate-2024\valid\SNGS-021"
    
    print(f"Initializing GameStateRecognizer...")
    print(f"Repo Dir: {repo_dir}")
    print(f"Video Path: {video_path}")
    
    recognizer = GameStateRecognizer(repo_dir=repo_dir)
    
    try:
        # We use fast_mode=True to skip ReID and OCR since they are extremely slow on standard hardware.
        # We also pass max_frames=3 to only process 3 frames, ensuring the test completes in a couple of minutes on CPU.
        print("\nStarting video processing (Fast Mode, 3 frames)...")
        result = recognizer.process_video(video_path, fast_mode=True, max_frames=30)
        
        print("\nProcessing complete! Output summary:")
        print(f"Total Frames Processed: {len(result.get('frames', {}))}")
        
        # Save output to JSON for inspection
        output_file = "test_output.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nFull structured output saved to {output_file}")
        
    except Exception as e:
        print(f"\nAPI Test Failed: {e}")

if __name__ == "__main__":
    test_api()
