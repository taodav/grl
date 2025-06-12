import sys
from scripts.batch_run_kitchen_sinks_single_obj import main

if __name__ == "__main__":
    # Add the "--config config.json" argument to sys.argv
    sys.argv.extend(["--config", "config.json"])
    
    # Call the main function from batch_run_kitchen_sinks_single_obj.py
    main()