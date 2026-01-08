#!/usr/bin/env python3
"""
Wrapper script to run the video detection benchmark. 
This MUST be the file you run:  python run.py
"""

if __name__ == "__main__":
    # Set multiprocessing start method BEFORE any imports
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Now import and run the main module
    from video_detection import main
    main()