import subprocess
import concurrent.futures

# Define the command as a function
times=6
def run_game():
    command = ["python3", "game.py", "ai2", "ai", "--dim", "4", "--time", "240"]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

# Run the command 10 times in parallel
if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=times) as executor:
        futures = [executor.submit(run_game) for _ in range(times)]
        
        # Wait for the results and print output
        for future in concurrent.futures.as_completed(futures):
            pass