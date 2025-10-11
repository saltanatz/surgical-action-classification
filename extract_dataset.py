"""
Extract RAR files from the dataset folder into organized structure.
This script extracts all .rar files into a 'clips' subdirectory.
"""

import os
import subprocess
import glob
from pathlib import Path

def extract_rar_files(source_dir, output_dir):
    """
    Extract all .rar files from source_dir into output_dir/clips/
    
    Args:
        source_dir: Directory containing .rar files
        output_dir: Directory where files will be extracted
    """
    # Convert to Path objects
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    clips_dir = output_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .rar files
    rar_files = sorted(source_path.glob("*.rar"))
    
    if not rar_files:
        print(f"❌ No .rar files found in {source_dir}")
        return
    
    print(f"Found {len(rar_files)} .rar files to extract")
    print(f"Source directory: {source_path}")
    print(f"Output directory: {clips_dir}")
    print("=" * 60)
    
    # Check if unrar is available
    try:
        subprocess.run(["unrar"], capture_output=True, check=False)
        unrar_cmd = "unrar"
    except FileNotFoundError:
        try:
            subprocess.run(["unar"], capture_output=True, check=False)
            unrar_cmd = "unar"
        except FileNotFoundError:
            print("❌ Error: Neither 'unrar' nor 'unar' command found!")
            print("\nPlease install one of them:")
            print("  macOS:   brew install unar")
            print("           or")
            print("           brew install unrar")
            print("  Linux:   sudo apt-get install unrar")
            print("           or")
            print("           sudo apt-get install unar")
            return
    
    print(f"Using extraction tool: {unrar_cmd}\n")
    
    # Extract each .rar file
    success_count = 0
    failed_files = []
    
    for idx, rar_file in enumerate(rar_files, 1):
        print(f"[{idx}/{len(rar_files)}] Extracting: {rar_file.name}")
        
        try:
            if unrar_cmd == "unrar":
                # unrar x -o+ "file.rar" "destination/"
                result = subprocess.run(
                    ["unrar", "x", "-o+", str(rar_file), str(clips_dir)],
                    capture_output=True,
                    text=True,
                    check=False
                )
            else:  # unar
                # unar -o "destination/" "file.rar"
                result = subprocess.run(
                    ["unar", "-o", str(clips_dir), str(rar_file)],
                    capture_output=True,
                    text=True,
                    check=False
                )
            
            if result.returncode == 0:
                print(f"  ✅ Success")
                success_count += 1
            else:
                print(f"  ❌ Failed")
                print(f"  Error: {result.stderr[:200]}")
                failed_files.append(rar_file.name)
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")
            failed_files.append(rar_file.name)
    
    # Copy CSV files to output directory
    print("\n" + "=" * 60)
    print("Copying CSV files...")
    csv_files = ["train.csv", "val.csv", "test.csv"]
    
    for csv_file in csv_files:
        src = source_path / csv_file
        dst = output_path / csv_file
        
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
            print(f"  ✅ Copied {csv_file}")
        else:
            print(f"  ⚠️  {csv_file} not found")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total .rar files: {len(rar_files)}")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    # Count extracted files
    extracted_files = list(clips_dir.rglob("*"))
    extracted_videos = [f for f in extracted_files if f.is_file() and f.suffix in ['.mp4', '.avi', '.mov', '.mkv']]
    
    print(f"\nExtracted files in {clips_dir}:")
    print(f"  Total items: {len(extracted_files)}")
    print(f"  Video files: {len(extracted_videos)}")
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 Dataset ready at: {output_path}")
    print(f"📁 Video clips in: {clips_dir}")
    
    return output_path


def main():
    """
    Main function to extract dataset.
    """
    print("=" * 60)
    print("SLAM Dataset RAR Extractor")
    print("=" * 60)
    
    # Configuration
    source_dir = "/Users/itsu/Desktop/deep_learning/28104782"
    output_dir = "/Users/itsu/Desktop/deep_learning/dataset"
    
    print(f"\n📦 Source: {source_dir}")
    print(f"📂 Output: {output_dir}")
    print()
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ Error: Source directory not found: {source_dir}")
        return
    
    # Confirm before proceeding
    print("This will extract all .rar files to the output directory.")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Extract files
    extract_rar_files(source_dir, output_dir)


if __name__ == "__main__":
    main()
