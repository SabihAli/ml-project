import os
import subprocess
import glob
import pandas as pd
from pathlib import Path

class GameStateRecognizer:
    def __init__(self, repo_dir: str):
        """
        Initializes the Game State Recognizer wrapper.
        
        Args:
            repo_dir: The path to the sn-gamestate repository directory where uv is configured.
        """
        self.repo_dir = Path(repo_dir).absolute()
        
    def process_video(self, sequence_dir: str, fast_mode: bool = False, max_frames: int = None) -> dict:
        """
        Processes a SoccerNet game state sequence directory and returns the game state structured data.
        
        Args:
            sequence_dir: Absolute path to the SoccerNet sequence directory (e.g., .../gamestate-2024/valid/SNGS-021).
            fast_mode: If True, disables heavy modules like OCR and ReID to speed up processing.
            max_frames: If set, restricts processing to this many frames (useful for CPU testing).
            
        Returns:
            Dictionary containing the parsed game state per frame.
        """
        seq_path = Path(sequence_dir).absolute()
        if not seq_path.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

        # E.g., Extract "SNGS-021" and "valid" from the path
        sequence_id = seq_path.name
        split = seq_path.parent.name
        
        experiment_name = f"api_run_{sequence_id}"
        
        # Build the command using hydra overrides.
        # We rely on the native soccernet_gs dataset loader which knows how to read the img1 folders.
        cmd = [
            "uv", "run", "tracklab", "-cn", "soccernet",
            f"dataset.eval_set='{split}'",
            f"dataset.vids_dict.{split}=['{sequence_id}']",
            f"experiment_name={experiment_name}",
            # Disable evaluation to prevent trackeval errors on truncated sequences
            "eval_tracking=False" 
        ]
        
        if max_frames is not None:
             cmd.append(f"dataset.nframes={max_frames}")
             
        if fast_mode:
            # Override pipeline to only run detection, reid, tracking, and calibration (skip Jersey)
            cmd.append("pipeline=[bbox_detector,reid,track,pitch,calibration]")
            
        print(f"Running TrackLab Pipeline on given video...")
        print("Command: " + " ".join(cmd))
        
        # Execute the process
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("TrackLab Error Output:\n", result.stderr)
            raise RuntimeError(f"TrackLab processing failed with exit code {result.returncode}")
            
        # Locate the output file. TrackLab saves under outputs/{experiment_name}/YYYY-MM-DD/HH-MM-SS/
        output_base_dir = self.repo_dir / "outputs" / experiment_name
        
        # Find the most recent run folder
        if not output_base_dir.exists():
            raise FileNotFoundError("TrackLab did not create the expected output directory.")
            
        # Get the latest state file (pklz)
        state_files = list(output_base_dir.rglob("states/*.pklz"))
        if not state_files:
             raise FileNotFoundError("Could not find the tracker_state.pklz file in the output directory.")
             
        # Sort by modification time to get the latest
        state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_state_file = state_files[0]
        
        print(f"Parsing output state: {latest_state_file}")
        return self._parse_tracker_state(latest_state_file)

    def _parse_tracker_state(self, pklz_path: Path) -> dict:
        """
        Parses the TrackLab generated pickle file into a clean dictionary.
        """
        import torch  # Required to unpickle torch tensors inside the dataframe
        # Load the dataframe. TrackLab states are pandas DataFrames saved as compressed pickles.
        df = pd.read_pickle(pklz_path)
        
        output = {
            "frames": {}
        }
        
        # Iterate over unique frames
        # Tracklab usually indexes by 'video_id', 'frame', 'image_id' etc.
        # The exact structure depends on the tracklab version, but typically it contains 'frame', 'bbox_ltwh', 'track_id', etc.
        
        if df.empty:
            return output
            
        # Group by image_id (which usually corresponds to the frame number)
        if 'image_id' in df.columns:
            groupby_col = 'image_id'
        elif 'frame' in df.columns:
            groupby_col = 'frame'
        else:
            # Fallback if structure is unexpected
            groupby_col = df.index.get_level_values('image_id') if 'image_id' in df.index.names else None

        if groupby_col is not None:
             grouped = df.groupby(groupby_col)
        else:
             print("Warning: Could not find frame/image_id column. Returning raw dict.")
             return df.to_dict(orient='records')

        for frame_idx, group in grouped:
            frame_data = []
            for _, row in group.iterrows():
                # Extract relevant fields, handling missing columns gracefully
                detection = {
                    "track_id": int(row.get('track_id', -1)) if pd.notna(row.get('track_id')) else None,
                    "bbox_ltwh": row.get('bbox_ltwh', None),
                    "bbox_pitch_xy": row.get('bbox_pitch_xy', None),
                    "team": row.get('team', None),
                    "role": row.get('role', None),
                    "jersey_number": row.get('jersey_number', None)
                }
                
                # Clean up NaN values to None for valid JSON
                detection = {k: (v if pd.notna(v) else None) for k, v in detection.items() if v is not None}
                frame_data.append(detection)
                
            output["frames"][str(frame_idx)] = frame_data
            
        return output

# Example usage (can be removed later):
if __name__ == "__main__":
    # Point to the cloned sn-gamestate repository
    repo_dir = r"C:\Users\sabih\OneDrive\Desktop\game_state_recognition\sn-gamestate"
    
    recognizer = GameStateRecognizer(repo_dir=repo_dir)
    
    # Example video path (you should replace this with a real video path when testing)
    test_video = r"C:\Users\sabih\OneDrive\Desktop\game_state_recognition\data\SoccerNetGS\gamestate-2024\valid\SNGS-021\video.mp4"
    
    if os.path.exists(test_video):
        try:
            print("Testing Fast Mode...")
            result = recognizer.process_video(test_video, fast_mode=True)
            print("Successfully processed video. Output keys:", result.keys())
            
            # Print first frame data as an example
            first_frame = next(iter(result['frames'].values()), None)
            if first_frame:
                 print("Sample detection from first frame:", first_frame[0])
        except Exception as e:
            print("Test failed:", e)
    else:
        print(f"Test video not found: {test_video}. Please create a test script with a valid video path.")
